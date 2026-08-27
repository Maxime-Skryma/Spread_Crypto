# Spread_Crypto
First test

git clone https://github.com/binance/binance-public-data.git

cd /home/onyxia/work/binance-public-data/python
pip install -r requirements.txt
python download-kline.py -t spot -s BTCUSDT ETHUSDT SOLUSDT -i 1h \
    -startDate 2025-01-01 -endDate 2025-06-30 -skip-daily 1

