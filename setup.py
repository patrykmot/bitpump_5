from setuptools import setup

setup(
    name="Bitpump",
    version="0.5",
    description="This project predict stock market prices with AI",
    author="Patryk Motyczyński",
    author_email="patryk.motyczynski.eu@email.com",
    url="",
    packages=["bitpump"],
    install_requires=[
        "yfinance==0.2.31",
        "pandas==2.1.1",
        "torch==2.1.0",
        "pytest==7.4.2"
    ]
)
