import pandas as pd

tx = pd.read_csv("data/train_transaction.csv",
                 usecols=["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1", "addr1", "P_emaildomain"])
idn = pd.read_csv("data/train_identity.csv", usecols=["TransactionID", "DeviceType", "DeviceInfo"])
print("transactions:", tx.shape, " fraud rate:", round(tx.isFraud.mean(), 4))
print("identity rows:", idn.shape)
print("days covered:", round((tx.TransactionDT.max() - tx.TransactionDT.min()) / 86400))
print("OK" if tx.shape[0] > 500000 and 0.03 < tx.isFraud.mean() < 0.04 else "CHECK: numbers look different from expected (~590540 rows, ~3.5% fraud)")
