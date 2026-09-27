import ast
import copy
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent


APPLE = "US0378331005"
MICROSOFT = "US5949181045"


def instrument(isin=APPLE, holding=False, value="0", complete=True):
    return {
        "instrument_id": isin, "isin": isin, "is_holding": holding,
        "quantity": "2", "pending_quantity": "0",
        "current_position_value_eur": value, "trade_data_complete": complete,
    }


def evaluation(isin=APPLE, score=95):
    return {
        "instrument_id": isin, "score": score, "forward_metrics_summary": "Metrics",
        "bull_case": "Growth", "bear_case": "Valuation", "verdict": "HOLD",
        "confidence_notes": "Incomplete evidence",
    }


def preview(side="buy"):
    return {
        "result": {
            "intent": {"side": side.upper(), "isin": APPLE, "amount": "100", "shares": "2"},
            "pre_trade_checks_passed": True, "order_submission": {"submitted": False},
        },
        "confirmation": {"id": "confirm-123", "requires_accept_unsuitable": False},
        "presentation": {
            "section_order": ["costs"], "required_leaf_paths": ["costs.total"],
            "sections": {"costs": {"title": "Costs", "fields": [
                {"path": "costs.total", "label": "Total costs", "value": "1.00"},
            ]}},
        },
        "compliance": {
            "must_present_all_information": True,
            "requires_explicit_user_confirmation_between_phases": True,
            "forbid_automatic_phase_2_execution": True,
            "confirmation_must_be_separate_step": True,
        },
    }


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.settings = agent.load_settings()
        self.snapshot = agent.PortfolioSnapshot(
            Decimal("200"), Decimal("150"), Decimal("800"), Decimal("1000"), []
        )

    def test_configuration_rejects_overlapping_or_missing_bands(self):
        source = agent.CONFIG_PATH.read_text()
        for changed in (
            source.replace("min_score = 90", "min_score = 89"),
            source.replace("min_score = 90", "min_score = 91"),
            source.replace("target_position_pct = 10", "target_position_pct = 11"),
        ):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                config = Path(folder) / "config.toml"
                config.write_text(changed)
                with self.assertRaises(agent.ConfigurationError):
                    agent.load_settings(config)

    def test_decimal_validation(self):
        for value in (True, None, "NaN", "Infinity", "-Infinity", "bad"):
            with self.subTest(value=value), self.assertRaises(agent.AgentError):
                agent.to_decimal(value, "test")
        self.assertEqual(agent.decimal_text(Decimal("1E-7")), "0.0000001")

    def test_buy_sizing_boundaries_cash_and_rounding(self):
        for score, current, cash, expected in (
            (85, "0", "500", "0"), (86, "0", "500", "30"),
            (90, "0", "500", "50"), (95, "0", "500", "100"),
            (100, "80.001", "500", "19.99"), (100, "100", "500", "0"),
            (100, "0", "12.349", "12.34"), (100, "0", "9.99", "0"),
        ):
            with self.subTest(score=score, current=current, cash=cash):
                amount, _ = agent.calculate_buy_amount(
                    score=score, current_position_value=Decimal(current),
                    total_portfolio_value=Decimal("1000"), available_cash=Decimal(cash),
                    settings=self.settings,
                )
                self.assertEqual(amount, Decimal(expected))

    def test_snapshot_includes_cash_and_pending_buys(self):
        snapshot = agent.build_portfolio_snapshot(
            {"valuation": {"total": "800"}},
            {"cash_balance": "200", "buying_power_without_credit": "150"},
            {"items": [{"isin": APPLE, "quantity": "2", "pending_quantity": "1",
                        "quote_mid_price": "25", "valuation": "60"}]},
        )
        self.assertEqual(snapshot.total_value, Decimal("1000"))
        self.assertEqual(snapshot.available_cash, Decimal("150"))
        self.assertEqual(snapshot.holdings[0]["current_position_value_decimal"], Decimal("85"))

    def test_plan_reserves_cash_in_score_order_without_sale_proceeds(self):
        instruments = [instrument(APPLE), instrument(MICROSOFT)]
        analysis = {"evaluations": [evaluation(APPLE, 90), evaluation(MICROSOFT, 95)]}
        self.snapshot.available_cash = Decimal("120")
        plan = agent.build_trade_plan(analysis, instruments, self.snapshot, self.settings)
        self.assertEqual([(item.isin, item.amount) for item in plan.items], [
            (MICROSOFT, Decimal("100")), (APPLE, Decimal("20")),
        ])
        instruments[0]["is_holding"] = True
        analysis["evaluations"][0]["score"] = 10
        self.snapshot.available_cash = Decimal("0")
        plan = agent.build_trade_plan(analysis, instruments, self.snapshot, self.settings)
        self.assertEqual([item.side for item in plan.items], ["sell"])

    def test_incomplete_data_blocks_both_sides(self):
        instruments = [instrument(APPLE, holding=True, complete=False), instrument(MICROSOFT, complete=False)]
        analysis = {"evaluations": [evaluation(APPLE, 10), evaluation(MICROSOFT, 95)]}
        plan = agent.build_trade_plan(analysis, instruments, self.snapshot, self.settings)
        self.assertEqual(plan.items, [])
        self.assertEqual(len(plan.warnings), 2)

    def test_analysis_validation_and_policy_verdict(self):
        analysis = {"portfolio_summary": "Summary", "evaluations": [evaluation()]}
        validated = agent.validate_analysis(copy.deepcopy(analysis), [instrument()], self.settings)
        self.assertEqual(validated["evaluations"][0]["verdict"], "BUY")
        for evaluations in ([], [evaluation(), evaluation()], [evaluation(score=True)], [evaluation(score=101)], [evaluation("unknown")]):
            with self.subTest(evaluations=evaluations), self.assertRaises(agent.AnalysisError):
                agent.validate_analysis({**analysis, "evaluations": evaluations}, [instrument()], self.settings)

    def test_cli_envelope_timeout_and_arguments(self):
        with patch("agent.subprocess.run", return_value=SimpleNamespace(
            stdout=json.dumps({"ok": True, "data": {"result": {"items": []}}}), stderr="", returncode=0,
        )) as run:
            self.assertEqual(agent.get_portfolio_holdings(12), {"items": []})
            self.assertEqual(run.call_args.args[0], ["sc", "broker", "holdings", "--json"])
            self.assertEqual(run.call_args.kwargs["timeout"], 12)
        for payload in ({"result": {}}, {"ok": True}, {"ok": False, "error": {"code": "blocked", "message": "No"}}):
            with self.subTest(payload=payload), patch("agent.subprocess.run", return_value=SimpleNamespace(
                stdout=json.dumps(payload), stderr="", returncode=0,
            )), self.assertRaises(agent.ScalableCLIError):
                agent.run_sc_command(["broker", "overview"])
        with patch("agent.subprocess.run", side_effect=subprocess.TimeoutExpired("sc", 1)), self.assertRaises(agent.ScalableCLIError):
            agent.run_sc_command(["broker", "overview"], 1)

    def test_candidate_conflicting_isin_is_not_verified(self):
        with patch("agent.search_stock", return_value={"items": [{"isin": MICROSOFT}]}):
            resolved = agent._resolve_candidate_isin("AAPL", {"isin": APPLE}, self.settings)
        self.assertFalse(resolved[2])

    def test_duplicate_candidates_are_analyzed_once(self):
        data = {"data_complete": True, "quote_type": "EQUITY"}
        with patch("agent.get_stock_evaluation_data", return_value=data), patch(
            "agent._resolve_candidate_isin", return_value=("", {}, False, "unresolved")
        ):
            instruments = agent.build_instruments(self.snapshot, ["aapl", "AAPL"], self.settings)
        self.assertEqual(len(instruments), 1)

    def test_nonfinite_market_numbers_are_missing(self):
        for value in ("nan", float("inf"), "bad", None):
            self.assertIsNone(agent._clean_number(value))
        self.assertEqual(agent._clean_number(0, percentage=True), 0)

    def test_preview_validates_identity_size_and_compliance(self):
        item = agent.TradePlanItem("buy", APPLE, APPLE, 95, amount=Decimal("100"))
        agent.validate_trade_preview(preview(), item)
        for field, value in (("side", "SELL"), ("isin", MICROSOFT), ("amount", "101")):
            data = preview()
            data["result"]["intent"][field] = value
            with self.subTest(field=field), self.assertRaises(agent.ScalableCLIError):
                agent.validate_trade_preview(data, item)
        data = preview()
        data["compliance"]["forbid_automatic_phase_2_execution"] = False
        with self.assertRaises(agent.ScalableCLIError):
            agent.validate_trade_preview(data, item)

    def test_confirmation_requires_disclosure_yes_and_refresh(self):
        item = agent.TradePlanItem("buy", APPLE, APPLE, 95, amount=Decimal("100"))
        plan = agent.TradePlan([item], [])
        for execute, answer, missing_disclosure, changed_cash, expected in (
            (False, "YES", False, False, 0),
            (True, "no", False, False, 0),
            (True, "yes", False, False, 0),
            (True, "YES", True, False, 0),
            (True, "YES", False, True, 0),
            (True, "YES", False, False, 1),
        ):
            data = preview()
            if missing_disclosure:
                data["presentation"]["sections"]["costs"]["fields"] = []
            latest = copy.deepcopy(self.snapshot)
            if changed_cash:
                latest.available_cash = Decimal("10")
            with self.subTest(execute=execute, answer=answer, missing=missing_disclosure, changed=changed_cash), \
                 patch("agent.preview_buy_order", return_value=data), \
                 patch("agent.confirm_buy_order", return_value={}) as submit, \
                 patch("agent.fetch_portfolio_snapshot", return_value=latest) as refresh, \
                 patch("builtins.input", return_value=answer) as ask, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                agent.execute_trade_plan(plan, self.settings, execute)
                self.assertEqual(submit.call_count, expected)
                if not execute or missing_disclosure:
                    ask.assert_not_called()
                if expected:
                    refresh.assert_called_once()
                    submit.assert_called_once_with(
                        APPLE, Decimal("100"), "confirm-123", accept_unsuitable=False,
                        timeout_seconds=self.settings.sc_command_timeout_seconds,
                    )

    def test_sell_refresh_blocks_reduced_shares(self):
        item = agent.TradePlanItem("sell", APPLE, APPLE, 10, shares=Decimal("2"))
        with patch("agent.fetch_portfolio_snapshot", return_value=self.snapshot), self.assertRaises(agent.ScalableCLIError):
            agent.revalidate_before_confirmation(item, self.settings)

    def test_analysis_only_does_not_preview(self):
        instruments = [instrument()]
        analysis = {"portfolio_summary": "Summary", "evaluations": [evaluation()]}
        analysis = agent.validate_analysis(analysis, instruments, self.settings)
        with patch("agent.load_dotenv", Mock()), patch("agent.check_cli_capabilities"), \
             patch("agent.fetch_portfolio_snapshot", return_value=self.snapshot), \
             patch("agent.build_instruments", return_value=instruments), \
             patch("agent.analyze_instruments", return_value=analysis), \
             patch("agent.execute_trade_plan") as execute, redirect_stdout(io.StringIO()):
            self.assertEqual(agent.main(["AAPL", "--analysis-only"]), 0)
            execute.assert_not_called()

    def test_structured_analysis_request_has_no_trade_tools(self):
        client = Mock()
        client.responses.create.return_value.output_text = json.dumps({
            "portfolio_summary": "Summary", "evaluations": [evaluation()],
        })
        with patch("agent.OpenAI", return_value=client), patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            analysis = agent.analyze_instruments([instrument()], self.snapshot, self.settings)
        kwargs = client.responses.create.call_args.kwargs
        self.assertNotIn("tools", kwargs)
        self.assertFalse(kwargs["store"])
        self.assertEqual(kwargs["text"]["format"]["type"], "json_schema")
        self.assertEqual(analysis["evaluations"][0]["verdict"], "BUY")

    def test_all_imports_precede_definitions(self):
        tree = ast.parse(Path(agent.__file__).read_text())
        first_definition = min(node.lineno for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self.assertLess(node.lineno, first_definition)


if __name__ == "__main__":
    unittest.main()
