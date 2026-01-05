"""
This is a beancount importer for Interactive Brokers. 
Setup:
1) have a running beancount system
2) activate IB FLexQuery with the entries specified in []
3) in the config.py file, specify a file location wiht your IBKR FlexQuery 
    Credentials
4) run 'bean-extract config.py ibkr.yml -f mainLedgerFile.bean
"""

import pandas as pd
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import warnings
import pickle
import re
import numpy as np
import logging

import yaml
from os import path
from ibflex import client, parser, Types
from ibflex.enums import CashAction, BuySell
from ibflex.client import ResponseCodeError

from beancount.parser import options
from beangulp import importer
from beancount.core import data, amount
from beancount.core.number import D
from beancount.core.number import Decimal
from beancount.core import position
from beancount.core.number import MISSING
from beanquery import query


class IBKRImporter(importer.Importer):
    """
    Beancount Importer for the Interactive Brokers XML FlexQueries
    """

    def __init__(self,
                 Mainaccount=None,  # for example Assets:Invest:IB
                 currency='CHF',
                 divSuffix='Div',  # suffix for dividend Account , like Assets:Invest:IB:VT:Div
                 DividendsAccount=None,
                 interestSuffix='Interest',
                 WHTAccount=None,
                 FeesSuffix='Fees',
                 FeesAccount=None,
                 PnLSuffix='PnL',
                 fpath=None,  #
                 depositAccount='',
                 suppressClosedLotPrice=False,
                 symbolMap={},
                 configFile='ibkr.yaml'
                 ):

        self.Mainaccount = Mainaccount  # main IB account in beancount
        self.currency = currency        # main currency of IB account
        self.divSuffix = divSuffix
        self.DividendsAccount = DividendsAccount
        self.WHTAccount = WHTAccount
        self.interestSuffix = interestSuffix
        self.FeesSuffix = FeesSuffix
        self.FeesAccount = FeesAccount
        self.PnLSuffix = PnLSuffix
        self.filepath = fpath             # optional file path specification,
        # if flex query should not be used online (loading time...)
        # Cash deposits are usually already covered
        self.depositAccount = depositAccount
        # by checkings account statements. If you want anyway the
        # deposit transactions, provide a True value
        self.suppressClosedLotPrice = suppressClosedLotPrice
        self.flag = '*'
        self.symbolMap = symbolMap
        self.configFile = configFile
        self.roc_str = "Return of Capital" # that special swiss thing

    def identify(self, file):
        return self.configFile == path.basename(file)

    @property
    def name(self):
        return self.configFile

    def getLiquidityAccount(self, currency):
        # Assets:Invest:IB:USD
        return ':'.join([self.Mainaccount, currency])

    def mapSymbol(self, symbol):
        return self.symbolMap.get(symbol, symbol)

    def getDivIncomeAcconut(self, currency, symbol):
        if self.DividendsAccount:
            return self.DividendsAccount
        else:
            # Income:Invest:IB:VTI:Div
            return ':'.join([self.Mainaccount.replace('Assets', 'Income'),
                            self.mapSymbol(symbol), self.divSuffix])

    def getInterestIncomeAcconut(self, currency):
        # Income:Invest:IB:USD
        return ':'.join([self.Mainaccount.replace('Assets', 'Income'), self.interestSuffix, currency])

    def getAssetAccount(self, symbol):
        # Assets:Invest:IB:VTI
        return ':'.join([self.Mainaccount, self.mapSymbol(symbol)])

    def getWHTAccount(self, symbol):
        # Expenses:Invest:IB:VTI:WTax
        return ':'.join([self.WHTAccount, self.mapSymbol(symbol)])

    def getFeesAccount(self, currency):
        if self.FeesAccount:
            return self.FeesAccount
        else:
            # Expenses:Invest:IB:Fees:USD
            return ':'.join([self.Mainaccount.replace('Assets', 'Expenses'), self.FeesSuffix, currency])

    def getPNLAccount(self, symbol):
        # Expenses:Invest:IB:Fees:USD
        return ':'.join([self.Mainaccount.replace('Assets', 'Income'),
                        self.mapSymbol(symbol), self.PnLSuffix])

    def account(self, filepath):
        return self.Mainaccount

    def extract(self, filepath, existing=None):
        # the actual processing of the flex query

        # get the IBKR creentials ready
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                token = config['token']
                queryId = config['queryId']
        except:
            warnings.warn('cannot read IBKR credentials file. Check filepath.')
            return []

        # get prices of existing transactions, in case we sell something
        # priceLookup = PriceLookup(existing, config['baseCcy'])

        if self.filepath is None:
            # get the report from IB. might take a while, when IB is queuing due to
            # traffic
            try:
                # try except in case of connection interrupt
                # Warning: queries sometimes take a few minutes until IB provides
                # the data due to busy servers
                response = client.download(token, queryId)
                statement = parser.parse(response)
            except ResponseCodeError as E:
                logging.exception('Error fetching report, aborting')
                return []
            except Exception as E:
                warnings.warn(f'could not fetch IBKR Statement. exiting. {E}')
                # another option would be to try again
                return []
            assert isinstance(statement, Types.FlexQueryResponse)
        else:
            print('**** loading from pickle')
            with open(self.filepath, 'rb') as pf:
                statement = pickle.load(pf)

        # convert to dataframes
        print(f"IBKR Import Debug: Found {len(statement.FlexStatements)} FlexStatement(s)")

        # Process all FlexStatements, not just the first one
        all_cash_transactions = []
        all_trades = []
        all_cash_reports = []

        for i, flex_statement in enumerate(statement.FlexStatements):
            print(f"IBKR Import Debug: Processing FlexStatement {i+1}")

            # Collect data from each FlexStatement
            if hasattr(flex_statement, 'CashTransactions') and flex_statement.CashTransactions:
                cash_trans = [{key: val for key, val in entry.__dict__.items()}
                             for entry in flex_statement.CashTransactions]
                all_cash_transactions.extend(cash_trans)
                print(f"  CashTransactions: {len(cash_trans)}")

            if hasattr(flex_statement, 'Trades') and flex_statement.Trades:
                trades = [{key: val for key, val in entry.__dict__.items()}
                         for entry in flex_statement.Trades]
                all_trades.extend(trades)
                print(f"  Trades: {len(trades)}")

            if hasattr(flex_statement, 'CashReport') and flex_statement.CashReport:
                cash_rpt = [{key: val for key, val in entry.__dict__.items()}
                           for entry in flex_statement.CashReport]
                all_cash_reports.extend(cash_rpt)
                print(f"  CashReport: {len(cash_rpt)}")

        # Create DataFrames from all FlexStatements combined
        ct = pd.DataFrame(all_cash_transactions)
        tr = pd.DataFrame(all_trades)
        cr = pd.DataFrame(all_cash_reports)

        print(f"IBKR Import Debug: Total combined - CashTransactions: {len(ct)}, Trades: {len(tr)}, CashReport: {len(cr)}")

        # Store in tabs dict for compatibility with existing code
        tabs = {'CashTransactions': ct, 'Trades': tr, 'CashReport': cr}

        # get single dataFrames
        ct = tabs['CashTransactions']
        tr = tabs['Trades']
        cr = tabs['CashReport']

        # Check for raw data duplicates before processing
        if not ct.empty:
            print(f"IBKR Import Debug: Checking CashTransactions for duplicates...")
            ct_before = len(ct)
            # Check for exact duplicate rows
            ct_dups = ct.duplicated()
            if ct_dups.any():
                print(f"IBKR Import Debug: Found {ct_dups.sum()} exact duplicate rows in CashTransactions")
                ct = ct.drop_duplicates()
                print(f"IBKR Import Debug: Removed duplicates, CashTransactions reduced from {ct_before} to {len(ct)}")

            # Check for potential functional duplicates (same date, type, symbol, amount)
            if len(ct) > 0:
                key_cols = [col for col in ['reportDate', 'type', 'symbol', 'amount', 'currency'] if col in ct.columns]
                if len(key_cols) >= 3:  # Need at least some key columns to detect duplicates
                    ct_func_dups = ct.duplicated(subset=key_cols)
                    if ct_func_dups.any():
                        print(f"IBKR Import Debug: Found {ct_func_dups.sum()} potential functional duplicates based on key fields")

        # throw out IBKR jitter, mostly None
        ct.drop(columns=[col for col in ct if all(
            ct[col].isnull())], inplace=True)
        tr.drop(columns=[col for col in tr if all(
            tr[col].isnull())], inplace=True)
        cr.drop(columns=[col for col in cr if all(
            cr[col].isnull())], inplace=True)
        transactions = self.Trades(
            tr) + self.CashTransactions(ct) + self.Balances(cr)

        print(f"IBKR Import Debug: Generated {len(transactions)} total transactions before deduplication")

        # First, remove internal duplicates within this import
        unique_transactions = []
        duplicate_count = 0

        for i, txn in enumerate(transactions):
            is_internal_duplicate = False

            # Only check transactions, not balances
            if isinstance(txn, data.Transaction):
                for existing_txn in unique_transactions:
                    if (isinstance(existing_txn, data.Transaction) and
                        txn.date == existing_txn.date and
                        txn.narration == existing_txn.narration and
                        len(txn.postings) == len(existing_txn.postings)):

                        # Check if postings match
                        postings_match = True
                        for p1, p2 in zip(txn.postings, existing_txn.postings):
                            if (p1.account != p2.account or
                                p1.units != p2.units):
                                postings_match = False
                                break

                        if postings_match:
                            is_internal_duplicate = True
                            duplicate_count += 1
                            print(f"IBKR Import Debug: Removing internal duplicate - {txn.date} {txn.narration}")
                            break

            if not is_internal_duplicate:
                unique_transactions.append(txn)

        if duplicate_count > 0:
            print(f"IBKR Import Debug: Removed {duplicate_count} internal duplicates, {len(unique_transactions)} transactions remaining")

        # Filter out duplicate transactions if existing entries are provided
        if existing:
            new_transactions = []
            external_duplicate_count = 0
            for txn in unique_transactions:
                if not is_duplicate_transaction(txn, existing):
                    new_transactions.append(txn)
                else:
                    external_duplicate_count += 1

            if external_duplicate_count > 0:
                print(f"IBKR Import Debug: Filtered out {external_duplicate_count} transactions that already exist in ledger")
                print(f"IBKR Import Debug: Final result: {len(new_transactions)} new transactions to import")

            return new_transactions

        print(f"IBKR Import Debug: Final result: {len(unique_transactions)} transactions to import")
        return unique_transactions

    def CashTransactions(self, ct):
        """
        This function turns the cash transactions table into beancount transactions
        for dividends, Witholding Tax, Cash deposits (if the flag is set in the
        ConfigIBKR.py) and Interests.
        arg ct: pandas DataFrame with the according data
        returns: list of Beancount transactions 
        """
        if len(ct) == 0:  # catch case of empty dataframe
            return []

        # first, separate different sorts of Data
        # Cash dividend is split from payment in lieu of a dividend.
        # Match them accordingly with the corresponding wht rows.
        # Make a copy of dataframe prior to append a column to avoid SettingWithCopyWarning
        dist = ct[ct['type'].map(lambda t: t == CashAction.DIVIDEND
                                or t == CashAction.PAYMENTINLIEU)].copy()   # dividends only (both cash and payment in lieu of d.)
        
         # special swiss thing that looks like a dividend but legally isnt
        dist["roc"] = dist.description.str.contains(self.roc_str)
        div = dist[~dist.roc]
        roc = dist[dist.roc]
        # Duplicate column to match later with wht
        div['__divtype__'] = div['type']

        # Make a copy of dataframe prior to append a column to avoid SettingWithCopyWarning
        wht = ct[ct['type'] == CashAction.WHTAX].copy()              # WHT only

        # create pseudo colum __divtype__ to match to div's __divtype__
        wht['__divtype__'] = wht['description'].map(lambda d:
                                                    CashAction.PAYMENTINLIEU if re.match('.*payment in lieu of dividend', d, re.IGNORECASE)
                                                    else CashAction.DIVIDEND)

        if len(div) != len(wht):
            matches = self.Dividends(div, with_wht=False)
        elif len(div) == 0:
            # in case of no dividends,
            matches = []
        else:
            # matching WHT & div
            match = pd.merge(
                div, wht, on=['symbol', 'reportDate', '__divtype__'])
            matches = self.Dividends(match)
        matches.extend(self.Dividends(roc,with_wht=False))

        dep = ct[ct['type'] == CashAction.DEPOSITWITHDRAW]    # Deposits only
        if len(dep) > 0:
            deps = self.Deposits(dep)
        else:
            deps = []

        int_ = ct[ct['type'].map(lambda t: t == CashAction.BROKERINTRCVD
                                 or t == CashAction.BROKERINTPAID)]     # interest only
        if len(int_) > 0:
            ints = self.Interest(int_)
        else:
            ints = []

        fee = ct[ct['type'] == CashAction.FEES]  # Fees only
        if len(fee) > 0:
            fees = self.Fee(fee)
        else:
            fees = []
        # list of transactiosn with short name
        ctTransactions = matches + deps + ints + fees

        return ctTransactions

    def Fee(self, fee):
        # calculates fees from IBKR data
        feeTransactions = []
        for idx, row in fee.iterrows():
            currency = row['currency']
            amount_ = amount.Amount(row['amount'], currency)
            text = row['description']
            month = re.findall(r'\w{3} \d{4}', text)
            if month:
                month = month[0]
            else:
                month = text

            # make the postings, two for fees
            postings = [data.Posting(self.getFeesAccount(currency),
                                     -amount_, None, None, None, None),
                        data.Posting(self.getLiquidityAccount(currency),
                                     amount_, None, None, None, None)]
            meta = data.new_metadata(__file__, 0, {})  # actually no metadata
            feeTransactions.append(
                data.Transaction(meta,
                                 row['reportDate'],
                                 self.flag,
                                 'IB',     # payee
                                 ' '.join(['Fee', currency, month]),
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings))
        return feeTransactions

    def Dividends(self, match, with_wht=True):
        # this function crates Dividend transactions from IBKR data
        # make dividend & WHT transactions

        divTransactions = []
        for idx, row in match.iterrows():
            dx = row.get('description_x', '')
            symbol = self.mapSymbol(row['symbol'])
            if with_wht:
                currency = row['currency_x']
                currency_wht = row['currency_y']
                if currency != currency_wht:
                    warnings.warn(('Warning: Dividend currency {} ' +
                                   'mismatches WHT currency {}. Skipping this' +
                                   'Transaction').format(currency, currency_wht))
                    continue
                amount_div = amount.Amount(row['amount_x'], currency)
                amount_wht = amount.Amount(row['amount_y'], currency)
                text = dx
            else:
                currency = row['currency']
                amount_div = amount.Amount(row['amount'], currency)
                text = row['description']

            # Find ISIN in description in parentheses
            regex = "|".join([r'\(([a-zA-Z]{2}[a-zA-Z0-9]{9}\d)\)',
                              self.roc_str])
            if self.roc_str in text:
                isin = self.roc_str
            else:
                isin = re.findall(r'\(([a-zA-Z]{2}[a-zA-Z0-9]{9}\d)\)', text)[0]
            pershare_match = re.search(r'(\d*[.]\d*)(\D*)(PER SHARE)',
                                       text, re.IGNORECASE)
            # payment in lieu of a dividend does not have a PER SHARE in description
            pershare = pershare_match.group(1) if pershare_match else ''

            # make the postings, three for dividend/ wht transactions
            postings = [data.Posting(self.getDivIncomeAcconut(currency,
                                                              symbol),
                                     -amount_div, None, None, None, None),
                        ]
            if with_wht:
                postings.extend([
                        data.Posting(self.getWHTAccount(symbol),
                                     -amount_wht, None, None, None, None),
                        data.Posting(self.getLiquidityAccount(currency),
                                     AmountAdd(amount_div, amount_wht),
                                     None, None, None, None)
                        ])
            else:
                postings.append(
                        data.Posting(self.getLiquidityAccount(currency),
                                     amount_div, None, None, None, None)
                        )
            meta = data.new_metadata(
                'dividend', 0, {'isin': isin, 'per_share': pershare})
            in_lieu_flag = " in lieu" if re.match(
                '.*payment in lieu of dividend', dx, re.IGNORECASE) else ""
            divTransactions.append(
                data.Transaction(meta,  # could add div per share, ISIN,....
                                 row['reportDate'],
                                 self.flag,
                                 symbol,     # payee
                                 'Dividend '+symbol + in_lieu_flag,
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))

        return divTransactions

    def Interest(self, int_):
        # calculates interest payments from IBKR data
        intTransactions = []
        for idx, row in int_.iterrows():
            currency = row['currency']
            amount_ = amount.Amount(row['amount'], currency)
            text = row['description']
            month = re.findall(r'\w{3}-\d{4}', text)[0]

            # make the postings, two for interest payments
            # received and paid interests are booked on the same account
            postings = [data.Posting(self.getInterestIncomeAcconut(currency),
                                     -amount_, None, None, None, None),
                        data.Posting(self.getLiquidityAccount(currency),
                                     amount_, None, None, None, None)
                        ]
            meta = data.new_metadata('Interest', 0)
            intTransactions.append(
                data.Transaction(meta,  # could add div per share, ISIN,....
                                 row['reportDate'],
                                 self.flag,
                                 'IB',     # payee
                                 ' '.join(['Interest ', currency, month]),
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))
        return intTransactions

    def Deposits(self, dep):
        # creates deposit transactions from IBKR Data

        depTransactions = []
        # assumes you figured out how to deposit/ withdrawal without fees
        if len(self.depositAccount) == 0:  # control this from the config file
            return []
        for idx, row in dep.iterrows():
            currency = row['currency']
            amount_ = amount.Amount(row['amount'], currency)

            # make the postings. two for deposits
            postings = [data.Posting(self.depositAccount,
                                     -amount_, None, None, None, None),
                        data.Posting(self.getLiquidityAccount(currency),
                                     amount_, None, None, None, None)
                        ]
            meta = data.new_metadata('deposit/withdrawel', 0)
            depTransactions.append(
                data.Transaction(meta,  # could add div per share, ISIN,....
                                 row['reportDate'],
                                 self.flag,
                                 'self',     # payee
                                 "deposit / withdrawal",
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))
        return depTransactions

    def Trades(self, tr):
        """
        This function turns the IBKR Trades table into beancount transactions
        for Trades
        arg tr: pandas DataFrame with the according data
        returns: list of Beancount transactions 
        """
        if len(tr) == 0:  # catch the case of no transactions
            return []
        # forex transactions
        fx = tr[tr['symbol'].apply(isForex)]
        # Stocks transactions
        stocks = tr[~tr['symbol'].apply(isForex)]

        trTransactions = self.Forex(fx) + self.Stocktrades(stocks)

        return trTransactions

    def Forex(self, fx):
        # returns beancount transactions for IBKR forex transactions

        fxTransactions = []
        for idx, row in fx.iterrows():

            symbol = row['symbol']
            curr_prim, curr_sec = getForexCurrencies(symbol)
            currency_IBcommision = row['ibCommissionCurrency']
            proceeds = amount.Amount(round(row['proceeds'], 2), curr_sec)
            quantity = amount.Amount(round(row['quantity'], 2), curr_prim)
            price = amount.Amount(row['tradePrice'], curr_sec)
            commission = amount.Amount(
                round(row['ibCommission'], 2), currency_IBcommision)
            buysell = row['buySell'].name

            cost = position.CostSpec(
                number_per=None,
                number_total=None,
                currency=None,
                date=None,
                label=None,
                merge=False)

            postings = [
                data.Posting(self.getLiquidityAccount(curr_prim),
                             quantity, None, price, None, None),
                data.Posting(self.getLiquidityAccount(curr_sec),
                             proceeds, None, None, None, None),
                data.Posting(self.getLiquidityAccount(currency_IBcommision),
                             commission, None, None, None, None),
                data.Posting(self.getFeesAccount(currency_IBcommision),
                             minus(commission), None, None, None, None)
            ]

            fxTransactions.append(
                data.Transaction(data.new_metadata('FX Transaction', 0),
                                 row['tradeDate'],
                                 self.flag,
                                 symbol,     # payee
                                 ' '.join(
                                     [buysell, quantity.to_string(), '@', price.to_string()]),
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))
        return fxTransactions

    def Stocktrades(self, stocks):
        # return the stocks transactions

        stocktrades = stocks[stocks['levelOfDetail']
                             == 'EXECUTION']  # actual trades
        buy = stocktrades[(stocktrades['buySell'] == BuySell.BUY) |        # purchases, including cancelled ones
                          (stocktrades['buySell'] == BuySell.CANCELBUY)]   # and the cancellation transactions to keep balance
        sale = stocktrades[(stocktrades['buySell'] == BuySell.SELL) |      # sales, including cancelled ones
                           (stocktrades['buySell'] == BuySell.CANCELSELL)]  # and the cancellation transactions to keep balance
        # closed lots; keep index to match with sales
        lots = stocks[stocks['levelOfDetail'] == 'CLOSED_LOT']

        stockTransactions = self.Panic(sale, lots) + self.Shopping(buy)

        return stockTransactions

    def Shopping(self, buy):
        # let's go shopping!!

        Shoppingbag = []
        for idx, row in buy.iterrows():
            # continue # debugging
            currency = row['currency']
            currency_IBcommision = row['ibCommissionCurrency']
            symbol = self.mapSymbol(row['symbol'])
            proceeds = amount.Amount(round(row['proceeds'],2), currency)
            commission = amount.Amount(
                (round(row['ibCommission'],2)), currency_IBcommision)
            quantity = amount.Amount(row['quantity'], symbol)
            price = amount.Amount(round(row['tradePrice'],2), currency)
            text = row['description']

            number_per = D(row['tradePrice'])
            currency_cost = currency
            cost = position.CostSpec(
                number_per=price.number,
                number_total=None,
                currency=currency,
                date=row['tradeDate'],
                label=None,
                merge=False)

            postings = [
                data.Posting(self.getAssetAccount(symbol),
                             quantity, cost, None, None, None),
                data.Posting(self.getLiquidityAccount(currency),
                             proceeds, None, None, None, None),
                data.Posting(self.getLiquidityAccount(currency_IBcommision),
                             commission, None, None, None, None),
                data.Posting(self.getFeesAccount(currency_IBcommision),
                             minus(commission), None, None, None, None)
            ]

            Shoppingbag.append(
                data.Transaction(data.new_metadata('Buy', 0),
                                 row['dateTime'].date(),
                                 self.flag,
                                 symbol,     # payee
                                 ' '.join(
                                     ['BUY', quantity.to_string(), '@', price.to_string()]),
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))
        return Shoppingbag

    def Panic(self, sale, lots):
        # OMG, IT is happening!!

        Doom = []
        for idx, row in sale.iterrows():
            # continue # debugging
            currency = row['currency']
            currency_IBcommision = row['ibCommissionCurrency']
            symbol = row['symbol']
            proceeds = amount.Amount(round(row['proceeds'],2), currency)
            commission = amount.Amount(
                (round(row['ibCommission'],2)), currency_IBcommision)
            quantity = amount.Amount(row['quantity'], self.mapSymbol(symbol))
            price = amount.Amount(round(row['tradePrice'],2), currency)
            text = row['description']
            date = row['dateTime'].date()
            number_per = D(row['tradePrice'])
            currency_cost = currency

            # Closed lot rows (potentially multiple) follow sell row
            lotpostings = []
            sum_lots_quantity = 0
            # mylots: lots closed by sale 'row'
            # symbol must match; begin at the row after the sell row
            # we do not know the number of lot rows; stop iteration if quantity is enough
            mylots = lots[(lots['symbol'] == row['symbol'])
                          & (lots.index > idx)]
            for li, clo in mylots.iterrows():
                sum_lots_quantity += clo['quantity']
                if sum_lots_quantity > -row['quantity']:
                    # oops, too many lots (warning issued below)
                    break

                cost = position.CostSpec(
                    number_per=0 if self.suppressClosedLotPrice else round(
                        clo['tradePrice'], 2),
                    number_total=None,
                    currency=clo['currency'],
                    date=clo['openDateTime'].date(),
                    label=None,
                    merge=False)

                lotpostings.append(data.Posting(self.getAssetAccount(symbol),
                                                amount.Amount(-clo['quantity'], clo['symbol']), cost, price, None, None))

                if sum_lots_quantity == -row['quantity']:
                    # Exact match is expected:
                    # all lots found for this sell transaction
                    break

            if sum_lots_quantity != -row['quantity']:
                warnings.warn(f"Lots matching failure: sell index={idx}")

            postings = [
                # data.Posting(self.getAssetAccount(symbol),  # this first posting is probably wrong
                # quantity, None, price, None, None),
                data.Posting(self.getLiquidityAccount(currency),
                             proceeds, None, None, None, None)
            ] +  \
                lotpostings + \
                [data.Posting(self.getPNLAccount(symbol),
                              None, None, None, None, None),
                 data.Posting(self.getLiquidityAccount(currency_IBcommision),
                              commission, None, None, None, None),
                 data.Posting(self.getFeesAccount(currency_IBcommision),
                              minus(commission), None, None, None, None)
                 ]

            Doom.append(
                data.Transaction(data.new_metadata('Buy', 0),
                                 date,
                                 self.flag,
                                 self.mapSymbol(symbol),     # payee
                                 ' '.join(
                                     ['SELL', quantity.to_string(), '@', price.to_string()]),
                                 data.EMPTY_SET,
                                 data.EMPTY_SET,
                                 postings
                                 ))
        return Doom

    def Balances(self, cr):
        # generate Balance statements from IBKR Cash reports
        # balances
        crTransactions = []
        for idx, row in cr.iterrows():
            currency = row['currency']
            if currency == 'BASE_SUMMARY':
                continue  # this is a summary balance that is not needed for beancount
            amount_ = amount.Amount(row['endingCash'].__round__(2), currency)

            # make the postings. two for deposits
            postings = [data.Posting(self.depositAccount,
                                     -amount_, None, None, None, None),
                        data.Posting(self.getLiquidityAccount(currency),
                                     amount_, None, None, None, None)
                        ]
            meta = data.new_metadata('balance', 0)

            crTransactions.append(data.Balance(
                meta,
                row['toDate'] + timedelta(days=1),  # see tariochtools EC imp.
                self.getLiquidityAccount(currency),
                amount_,
                None,
                None))
        return crTransactions


def CollapseTradeSplits(tr):
    # to be implemented
    """
    This function collapses two trades into once if they have same date,symbol
    and trade price. IB sometimes splits up trades.
    """
    pass


def isForex(symbol):
    # retruns True if a transaction is a forex transaction.
    b = re.search(r"(\w{3})[.](\w{3})", symbol)  # find something lile "USD.CHF"
    if b == None:  # no forex transaction, rather a normal stock transaction
        return False
    else:
        return True


def getForexCurrencies(symbol):
    b = re.search(r"(\w{3})[.](\w{3})", symbol)
    c = b.groups()
    return [c[0], c[1]]


class InvalidFormatError(Exception):
    pass


def fmt_number_de(value: str) -> Decimal:
    # a fix for region specific number formats
    thousands_sep = '.'
    decimal_sep = ','

    return Decimal(value.replace(thousands_sep, '').replace(decimal_sep, '.'))


def DecimalOrZero(value):
    # for string to number conversion with empty strings
    try:
        return Decimal(value)
    except:
        return Decimal(0.0)


def AmountAdd(A1, A2):
    # add two amounts
    if A1.currency == A2.currency:
        quant = A1.number+A2.number
        return amount.Amount(quant, A1.currency)
    else:
        raise ('Cannot add amounts of differnent currencies: {} and {}'.format(
            A1.currency, A1.currency))


def is_duplicate_transaction(new_txn, existing_entries):
    """
    Check if a new transaction is a duplicate of an existing one.
    Compares by date, narration, and posting amounts.
    """
    # Only check Transaction objects, not Balance directives or other entry types
    if not isinstance(new_txn, data.Transaction):
        return False

    if not existing_entries:
        return False

    for existing in existing_entries:
        if not isinstance(existing, data.Transaction):
            continue

        # Compare basic transaction properties
        if (existing.date == new_txn.date and
            existing.narration == new_txn.narration and
            len(existing.postings) == len(new_txn.postings)):

            # Compare postings (account and amount)
            postings_match = True
            for existing_posting, new_posting in zip(existing.postings, new_txn.postings):
                if (existing_posting.account != new_posting.account or
                    existing_posting.units != new_posting.units):
                    postings_match = False
                    break

            if postings_match:
                return True

    return False


def minus(A):
    # a minus operator
    return amount.Amount(-A.number, A.currency)
