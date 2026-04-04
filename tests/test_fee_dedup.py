"""Tests for IBKR fee deduplication.

IB's flex query sometimes reports daily maintenance fees both as individual
per-day entries AND as a single combined entry covering the same period,
all sharing the same reportDate. After exact row dedup these still overlap:
e.g. 3 individual entries of 0.80 plus one combined entry of 2.40 on the
same reportDate.  The importer must keep only one representation.
"""

import datetime
from decimal import Decimal

import pandas as pd
import pytest
from ibflex.enums import CashAction

from drnukebean.importer.ibkr import IBKRImporter


@pytest.fixture
def importer():
    return IBKRImporter(
        Mainaccount="Assets:IB",
        currency="EUR",
        FeesAccount="Expenses:Fees:Broker",
    )


def _fee_row(report_date, amount, description="XETRA-GOLD MAINTENANCE FEE FOR 2026-03-27 FOR MAR 2026"):
    return {
        "type": CashAction.FEES,
        "currency": "EUR",
        "description": description,
        "amount": Decimal(str(amount)),
        "reportDate": report_date,
    }


class TestFeeDedup:
    """Individual + combined fee entries on the same date must not double-count."""

    def test_combined_and_individuals_keeps_combined(self, importer):
        """Three individual 0.80 entries + one 2.40 combined → single 2.40 transaction."""
        date = datetime.date(2026, 3, 30)
        rows = [
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-27 FOR MAR 2026"),
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-28 FOR MAR 2026"),
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-29 FOR MAR 2026"),
            _fee_row(date, Decimal("-2.40"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-27 FOR MAR 2026"),
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        total = sum(p.units.number for t in txns for p in t.postings if "Fees" in p.account)
        assert total == Decimal("2.40")

    def test_individuals_only_kept_intact(self, importer):
        """When there's no combined entry, individual entries are all kept."""
        date = datetime.date(2026, 3, 30)
        rows = [
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-27 FOR MAR 2026"),
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-28 FOR MAR 2026"),
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-29 FOR MAR 2026"),
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        total = sum(p.units.number for t in txns for p in t.postings if "Fees" in p.account)
        assert total == Decimal("2.40")

    def test_combined_only_kept_intact(self, importer):
        """A single combined entry with no individuals is kept as-is."""
        date = datetime.date(2026, 3, 30)
        rows = [
            _fee_row(date, Decimal("-2.40")),
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        assert len(txns) == 1
        total = sum(p.units.number for t in txns for p in t.postings if "Fees" in p.account)
        assert total == Decimal("2.40")

    def test_different_currencies_not_merged(self, importer):
        """Fees in different currencies on the same date stay separate."""
        date = datetime.date(2026, 3, 30)
        rows = [
            {**_fee_row(date, Decimal("-0.80")), "currency": "EUR"},
            {**_fee_row(date, Decimal("-0.80")), "currency": "USD"},
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        assert len(txns) == 2

    def test_unrelated_fees_same_date_not_merged(self, importer):
        """Different fee types on the same date are not merged."""
        date = datetime.date(2026, 3, 30)
        rows = [
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-27 FOR MAR 2026"),
            _fee_row(date, Decimal("-1.50"), "MARKET DATA FEE FOR MAR 2026"),
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        total = sum(p.units.number for t in txns for p in t.postings if "Fees" in p.account)
        assert total == Decimal("2.30")

    def test_two_individuals_plus_combined(self, importer):
        """Two individual 0.80 entries + one 1.60 combined → single 1.60 transaction."""
        date = datetime.date(2026, 3, 30)
        rows = [
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-28 FOR MAR 2026"),
            _fee_row(date, Decimal("-0.80"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-29 FOR MAR 2026"),
            _fee_row(date, Decimal("-1.60"), "XETRA-GOLD MAINTENANCE FEE FOR 2026-03-28 FOR MAR 2026"),
        ]
        fee_df = pd.DataFrame(rows)

        txns = importer.Fee(fee_df)
        total = sum(p.units.number for t in txns for p in t.postings if "Fees" in p.account)
        assert total == Decimal("1.60")
