import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from text2network.classes.neo4jnw import neo4j_network
from text2network.utils.file_helpers import check_create_folder
from text2network.utils.logging_helpers import setup_logger

# Set a configuration path
configuration_path = 'config/analyses/FounderSenBert40.ini'
# Settings
import os

os.environ['NUMEXPR_MAX_THREADS'] = '16'
alter_subset = None
# Load Configuration file
import configparser

config = configparser.ConfigParser()
print(check_create_folder(configuration_path))
config.read(check_create_folder(configuration_path))
# Setup logging
setup_logger(config['Paths']['log'], config['General']['logging_level'], "f_semantic_importance.py")

output_path = check_create_folder(config['Paths']['csv_outputs'])
output_path = check_create_folder(config['Paths']['csv_outputs'] + "/regression_tables/")
filename = check_create_folder("".join([output_path, "/founder_reg"]))

semantic_network = neo4j_network(config)


times = list(range(1980, 2021))

company_list = ["Apple", "Microsoft", "Alphabet (Google)", "Saudi Aramco", "Amazon", "Tesla", "Meta (Facebook)",
                "NVIDIA", "Berkshire Hathaway", "TSMC", "Tencent", "JPMorgan Chase", "Visa", "Samsung", "UnitedHealth",
                "Johnson & Johnson", "Home Depot", "Kweichow Moutai", "LVMH", "Walmart", "Nestle", "Procter & Gamble",
                "Bank of America", "Roche", "Alibaba", "Mastercard", "ASML", "Adobe", "Pfizer", "Walt Disney",
                "Netflix", "Nike", "Exxon Mobil", "L'oreal", "Salesforce", "Novo Nordisk", "Toyota",
                "Thermo Fisher Scientific", "ICBC", "Oracle", "Broadcom", "Cisco", "Coca-Cola", "Accenture", "Costco",
                "Abbott Laboratories", "Eli Lilly", "CATL", "PayPal", "Pepsico", "Chevron", "Danaher", "Comcast",
                "Reliance Industries", "AbbVie", "Verizon", "CM Bank", "Intel", "QUALCOMM", "Wells Fargo", "McDonald",
                "Hermes", "Meituan", "Shopify", "Intuit", "Merck", "Novartis", "Morgan Stanley", "Texas Instruments",
                "United Parcel Service", "Nextera Energy", "Tata Consultancy Services", "AMD", "Prosus",
                "Lowe's Companies", "Shell", "China Construction Bank", "Astrazeneca", "Linde", "AT&T",
                "Union Pacific Corporation", "SAP", "Agricultural Bank of China", "Keyence", "Sony", "Wuliangye Yibin",
                "Charles Schwab", "Medtronic", "BHP Group", "Royal Bank Of Canada", "Sea (Garena)", "T-Mobile US",
                "Dior", "Applied Materials", "Ping An Insurance", "Honeywell", "BlackRock", "Philip Morris", "Siemens",
                "Toronto Dominion Bank", "BYD", "PetroChina", "Starbucks", "Unilever", "Goldman Sachs", "ServiceNow",
                "TotalEnergies", "American Express", "Bank of China", "Raytheon Technologies", "AIA", "Estee Lauder",
                "Bristol-Myers Squibb", "Volkswagen", "American Tower", "HDFC Bank", "Boeing", "Citigroup", "Diageo",
                "CVS Health", "Jingdong Mall", "China Mobile", "Intuitive Surgical", "Sanofi", "Amgen",
                "Commonwealth Bank", "HSBC", "Airbnb", "Prologis", "Charter Communications", "Moderna", "Target",
                "S&P Global", "Snowflake", "Deere & Company", "Merck KGaA", "Macquarie", "China Life Insurance",
                "Caterpillar", "IBM", "Gazprom", "Zoetis", "General Electric", "Rivian", "Transurban",
                "GlaxoSmithKline", "Schneider Electric", "Recruit", "Daimler", "Rio Tinto", "Inditex", "3M", "Anthem",
                "Lam Research", "Anheuser-Busch Inbev", "CSL", "Kering", "Automatic Data Processing",
                "Stryker Corporation", "Infosys", "Atlassian", "Analog Devices", "Nippon Telegraph & Telephone",
                "ConocoPhillips", "Blackstone Group", "Micron Technology", "Allianz", "Airbus", "Booking.com",
                "Lockheed Martin", "EssilorLuxottica", "Canadian National Railway", "Brookfield Asset Management", "BP",
                "Deutsche Telekom", "Square", "Al Rajhi Bank", "Sherwin-Williams", "TJX Companies", "General Motors",
                "Compagnie FinanciÃ¨re Richemont", "Gilead Sciences", "Marsh & McLennan Companies", "Adyen", "Equinor",
                "SABIC", "Tokyo Electron", "Sberbank", "Snap", "Siemens Healthineers", "PNC Financial Services",
                "Mondelez", "U.S. Bancorp", "CME Group", "SoftBank", "Air Liquide", "Crown Castle", "Scotiabank",
                "CSX Corporation", "Altria Group", "British American Tobacco", "Chubb", "Atlas Copco",
                "Dassault SystÃ¨mes", "BNP Paribas", "Hikvision", "Truist Financial", "Midea", "Ford", "Pinduoduo",
                "Enbridge", "Investor AB", "Foshan Haitian Flavouring and Food", "Duke Energy",
                "Hong Kong Exchanges & Clearing", "Marvell Technology Group", "Enel", "Intercontinental Exchange",
                "Postal Savings Bank of China", "Sinopec", "Illinois Tool Works", "Uber", "Coinbase", "Deutsche Post",
                "Mitsubishi UFJ Financial", "HCA Healthcare", "Edwards Lifesciences", "Rosneft", "Moody's", "Petrobras",
                "Lucid Motors", "National Commercial Bank", "ABB", "Equinix", "Hindustan Unilever",
                "LONGi Green Energy Technology", "Shin-Etsu Chemical", "NetEase", "SK Hynix", "Roblox", "Iberdrola",
                "Mindray", "BioNTech", "Bank of Montreal", "Workday", "Regeneron Pharmaceuticals",
                "China Tourism Group Duty Free", "China Yangtze Power", "Becton Dickinson", "Norfolk Southern", "AXA",
                "ICICI Bank", "Novatek", "Cigna", "Fiserv", "Southern Company", "Nidec", "Waste Management", "Vale",
                "Nongfu Spring", "Housing Development Finance Corporation", "Eaton", "Daikin",
                "National Australia Bank", "WuXi AppTec", "Great Wall Motors", "Ecolab", "BMW", "KDDI", "Aon",
                "Fidelity National Information Services", "Colgate-Palmolive", "Air Products and Chemicals", "Ferrari",
                "FedEx", "Glencore", "China Shenhua Energy", "KLA", "Industrial Bank", "BASF", "Pernod Ricard",
                "Capital One", "Bank Central Asia", "Zurich Insurance Group", "MediaTek", "UBS", "RELX",
                "Fast Retailing", "Xiaomi", "America Movil", "Walmex", "MercadoLibre", "Lonza", "Dominion Energy",
                "Autodesk", "DBS", "East Money Information", "NXP Semiconductors", "Public Storage",
                "United Heavy Machinery", "Maersk", "Bajaj Finance", "Thomson Reuters", "Boston Scientific",
                "Stellantis", "Infineon", "Heineken", "Illumina", "Reckitt Benckiser", "Denso", "Lukoil", "Adidas",
                "Hoya", "Zoom", "EQT", "DoorDash", "Sika", "China Telecom", "State Bank of India", "Verbund AG",
                "Northrop Grumman", "Ping An Bank", "Datadog", "Freeport-McMoRan", "Humana", "General Dynamics",
                "Johnson Controls", "Hitachi", "ANZ Bank", "Synopsys", "NIO", "Xilinx", "Vinci", "WuXi Biologics",
                "Progressive", "Bharti Airtel", "Volvo Group", "Nintendo", "lululemon athletica", "Emerson",
                "Chugai Pharmaceutical", "DexCom", "Oriental Land", "Westpac Banking", "ING", "Ã˜rsted", "MSCI",
                "Align Technology", "Anglo American", "Foxconn", "Santander", "IHS Markit", "Exelon Corporation",
                "Fortinet", "EOG Resources", "Vertex Pharmaceuticals", "IDEXX Laboratories", "Palo Alto Networks",
                "IQVIA", "DSV", "TE Connectivity", "Baidu", "Jiangsu Hengrui Medicine", "Simon Property Group",
                "Safran", "Cloudflare", "Samsung Biologics", "Marriott International", "Dollar General",
                "CITIC Securities", "Bayer", "Cadence Design Systems", "MetLife", "Wanhua Chemical", "S.F. Express",
                "Naver", "Canadian Natural Resources", "CIBC", "Luxshare Precision", "The Trade Desk",
                "London Stock Exchange", "Kotak Mahindra Bank", "Amphenol", "ENI", "National Grid",
                "Roper Technologies", "Intesa Sanpaolo", "Liberty Bancshares", "Keurig Dr Pepper",
                "Microchip Technology", "Bank of Communications", "Coupang", "Twilio", "Chipotle Mexican Grill",
                "Carrier", "Murata Seisakusho", "Honda", "Z Holdings", "Wesfarmers", "Prudential", "Vmware",
                "Canadian Pacific Railway", "Givaudan", "CrowdStrike", "Wipro", "Trane Technologies", "Digital Realty",
                "Carvana", "EDF", "Agilent Technologies", "Sumitomo Mitsui Financial Group", "BCE", "Monster Beverage",
                "LG Chem", "Nordea Bank", "Cintas", "Southern Copper", "Spotify", "Bank of New York Mellon",
                "ItÅchÅ« ShÅji", "Activision Blizzard", "Kimberly-Clark", "Aptiv", "Enterprise Products",
                "T. Rowe Price", "Kuaishou Technology", "TC Energy", "American International Group", "Nornickel",
                "O'Reilly Automotive", "Ambev", "Pioneer Natural Resources", "Kakao", "KKR & Co.", "Newmont",
                "Jardine Matheson", "SMC", "Dell", "STMicroelectronics", "Unity Software", "Daiichi SankyÅ", "Paychex",
                "Anta Sports", "CNOOC", "YANGHE", "Banco Santander Brasil", "Lloyds Banking Group", "Hapag-Lloyd",
                "COSCO Shipping", "Constellation Brands", "Bank Rakyat Indonesia", "Zscaler", "Takeda Pharmaceutical",
                "3i Group", "Experian", "Schlumberger", "Partners Group", "Cognizant Technology Solutions", "Centene",
                "Republic Services", "Motorola Solutions", "L3Harris Technologies", "Muyuan Foods", "CrÃ©dit Agricole",
                "HP", "Kraft Heinz", "eBay", "Alimentation Couche-Tard", "HCL Technologies", "Walgreens Boots Alliance",
                "Veeva Systems", "Barclays", "Alcon", "American Electric Power", "AutoZone", "SVB Financial Group",
                "Fanuc", "Techtronic Industries", "Sartorius", "DMart", "Bank of Ningbo", "Vodafone", "Zijin Mining",
                "XPeng", "Old Dominion Freight Line", "Munich RE", "Samsung SDI", "Parker-Hannifin", "Dow",
                "Dupont De Nemours", "Shanghai Pudong Development Bank", "Rockwell Automation", "Cellnex Telecom",
                "Asian Paints", "Baxter", "Nutrien", "Sempra Energy", "Fortescue", "Hilton Worldwide", "Hyundai",
                "Ross Stores", "SAIC Motor", "Hexagon", "DSM", "Prudential Financial", "Capgemini", "Palantir",
                "EPAM Systems", "Yili Group", "Ashtead", "Marathon Petroleum", "ItaÃº Unibanco",
                "Cement Roadstone Holding", "7-Eleven", "PPG Industries", "ResMed", "SBA Communications",
                "AIER Eye Hospital", "Realty Income", "China Pacific Insurance", "General Mills",
                "Banco Bilbao Vizcaya Argentaria", "Yum! Brands", "Arista Networks", "ITC", "Match Group", "Danone",
                "First Republic Bank", "D. R. Horton", "The Travelers Companies", "Global Payments",
                "The Hershey Company", "International Flavors & Fragrances", "Sysco", "Aflac", "Neste", "OCBC Bank",
                "Okta", "Welltower", "Bajaj Finserv", "Compass Group", "Verisk Analytics", "Mitsui Bussan",
                "Foxconn Industrial Internet", "Tokio Marine", "Kinder Morgan", "Japan Tobacco", "HubSpot", "Keysight",
                "Manulife Financial", "Constellation Software", "Otis Worldwide", "Twitter", "Mettler-Toledo",
                "Banco Bradesco", "Fastenal", "Xcel Energy", "KBC", "Suncor Energy", "KÃ¼hne + Nagel", "ENGIE", "Vanke",
                "Equifax", "Sun Hung Kai Properties", "Affirm", "AppLovin", "KONE", "Archer Daniels Midland",
                "Ferguson", "Compagnie de Saint-Gobain", "Electronic Arts", "Vonovia", "Hang Seng Bank", "Copart",
                "MongoDB", "Straumann", "Waste Connections", "Ansys", "Woolworths Group", "DNB",
                "Arthur J. Gallagher & Co.", "Corteva", "Coloplast", "GlobalFoundries", "Larsen &amp", "Lennar",
                "Budweiser APAC", "Nasdaq", "Ahold Delhaize", "Sunny Optical", "McKesson", "Biogen", "Roku",
                "CBRE Group", "Gree Electric Appliances", "Telstra", "AvalonBay Communities", "Ericsson", "Naspers",
                "Fujitsu", "State Street Corporation", "Henkel", "Grupo MÃ©xico", "TransDigm",
                "Bank of China (Hong Kong)", "Ameriprise Financial", "Williams Companies", "DiDi",
                "Formosa Petrochemical", "Vestas Wind Systems", "E.ON", "Discover Financial Services", "NatWest Group",
                "Sydney Airport", "Generali", "Barrick Gold", "Brown Forman", "UOB", "Assa Abloy", "MTR Corporation",
                "Nokia", "Alexandria Real Estate Equities", "Equity Residential", "Corning", "Ametek", "Li Auto",
                "Sandvik", "Rocket Companies", "Zebra Technologies", "Fubon Financial", "PSEG", "West Pharma", "Nucor",
                "Philips", "Interactive Brokers", "Sun Life Financial", "American Water Works",
                "Mizuho Financial Group", "SMIC", "PTT PCL", "Longfor Group", "Chunghwa Telecom", "Terumo", "UniCredit",
                "Sany Heavy Industry", "Telus", "Japan Post Bank", "Kroger", "Cummins", "Skandinaviska Enskilda Banken",
                "Phillips 66", "Stanley Black &amp", "Allstate", "China Securities", "Delivery Hero", "CoStar Group",
                "Etsy", "Dollar Tree", "Bridgestone", "Anhui Conch Cement", "Albemarle", "Legrand", "Paccar",
                "Apollo Global Management", "Lindt", "Geely", "Fujifilm", "BOE Technology", "Tyson Foods",
                "Enphase Energy", "Ball Corporation", "DocuSign", "Devon Energy", "H&M", "Wolters Kluwer",
                "VF Corporation", "MPLX", "LyondellBasell", "Eversource Energy", "Holcim", "Amadeus IT Group",
                "NTT Data", "United Microelectronics", "Maruti Suzuki India", "Singtel", "Las Vegas Sands",
                "Weyerhaeuser", "BeiGene", "Flutter Entertainment", "JD Health", "Deutsche BÃ¶rse", "Porsche SE",
                "Willis Towers Watson", "Japan Post Holdings", "Peopleâ€™s Insurance Company of China",
                "WEC Energy Group", "Astellas Pharma", "Shenzhen Inovance", "Ecopetrol", "Valero Energy",
                "Cathay Financial Holding", "Consolidated Edison", "Epiroc", "Tesco", "Adani Green Energy",
                "Schindler Group", "Geberit", "Occidental Petroleum", "Kia", "Swisscom", "Olympus", "Michelin",
                "NIBE Industrier", "Pacific Gas and Electric", "LabCorp", "UltraTech Cement", "Hong Kong and China Gas",
                "Telkom Indonesia", "Axis Bank", "Extra Space Storage", "Orange", "Swiss Re", "Mitsubishi Electric",
                "Central Japan Railway", "Titan Company", "Evergreen Marine", "Oneok", "Great-West Lifeco",
                "ON Semiconductor", "ArcelorMittal", "SociÃ©tÃ© GÃ©nÃ©rale", "Liberty Broadband",
                "China Overseas Land &amp", "Naturgy", "China Resources Beer", "Adani Transmission", "Seagen", "WEG ON",
                "Sampo", "Fortum", "Fortive", "Bill.com", "Kansas City Southern", "Deutsche Bank", "CDW Corporation",
                "VeriSign", "Evolution Gaming", "Zimmer Biomet", "Kubota", "Southwest Airlines", "Martin Marietta",
                "Tractor Supply", "Cheniere Energy", "China Everbright Bank", "Garmin", "Sysmex", "RWE", "Nippon Paint",
                "Vulcan Materials", "ZoomInfo", "Saudi Electricity", "Skyworks Solutions", "Pinterest", "KE Holdings",
                "Airports of Thailand", "Shimano", "Genmab", "Energy Transfer Partners", "Gartner", "Paycom",
                "Teradyne", "Expedia", "W. W. Grainger", "National Bank of Canada", "Nomura Research Institute",
                "Synchrony", "United Rentals", "Invitation Homes", "Franco-Nevada", "Huazhu Hotels", "Fresenius",
                "Best Buy", "CRRC", "ZTO Express", "Loblaws", "Cenovus Energy", "Fomento EconÃ³mico Mexicano", "KaÅ",
                "Edison International", "Teleperformance", "Panasonic", "China Minsheng Bank", "Unicharm",
                "Delta Electronics", "AmerisourceBergen", "Toyota Industries", "Mid-America Apartment Communities",
                "Chewy", "Renesas Electronics", "Adani Enterprises", "Volvo Car", "Tata Motors", "Hess",
                "Coca-Cola European Partners", "Delta Air Lines", "Wayfair", "Monolithic Power Systems", "Sirius XM",
                "Live Nation", "CK Hutchison Holdings", "Dover", "Lasertec", "Northern Trust", "NestlÃ© India",
                "Oil &amp", "Celltrion", "Ingersoll Rand", "UiPath", "Imperial Oil", "Canon", "Avantor",
                "Nan Ya Plastics", "SGS", "Magna International", "Polyus", "CNH Industrial", "CLP Group",
                "EDP RenovÃ¡veis", "Shiseido", "ORIX", "Sun Pharmaceutical", "Formosa Plastics", "CarMax",
                "Galaxy Entertainment", "Yandex", "PerkinElmer", "TelefÃ³nica", "Church & Dwight", "East Japan Railway",
                "McCormick & Company", "Maybank", "Sun Communities", "Endesa", "Horizon Therapeutics", "Weichai Power",
                "Duke Realty", "Seagate Technology", "BTG Pactual", "Credit Suisse", "Beiersdorf", "Kakao Pay",
                "BAE Systems", "China Merchants Securities", "The Hartford", "Rogers Communication", "Hormel Foods",
                "Generac Power Systems", "Sonova", "SSE", "Bilibili", "Maaden", "EnBW Energie", "Carlsberg", "Steris",
                "Bio-Rad Laboratories", "Swedbank", "Veolia", "Ryanair", "BT Group", "Bank Mandiri", "Mengniu Dairy",
                "Carnival", "Huntington Bancshares", "Komatsu", "Essex Property Trust", "ICON plc", "POOLCORP",
                "Kyocera", "Tradeweb", "Power Corporation of Canada", "Dai-ichi Life Holdings", "Intact Financial",
                "Country Garden", "TransUnion", "Xylem", "Ameren", "Huatai Securities", "Essity", "ULTA Beauty", "Aena",
                "Warner Music Group", "DTE Energy", "Trimble", "Yum China", "Cerner", "Hannover RÃ¼ck", "ZTE",
                "Zalando", "Novozymes", "PPL", "EDP Group", "Catalent", "Expeditors", "CK Asset Holdings",
                "Tyler Technologies", "Baker Hughes", "Fortis", "Riyad Bank", "Regions Financial", "JSW Steel",
                "Aristocrat", "KeyCorp", "Saudi Arabian Fertilizer Company", "Continental", "CaixaBank", "FirstEnergy",
                "Halliburton", "UCB", "Entergy", "Henderson Land Development", "Ferrovial", "Alnylam Pharmaceuticals",
                "Kellogg's", "Waters Corporation", "CGI", "Telenor", "FirstRand", "Plug Power", "Ã†on", "Exor",
                "Afterpay", "Moncler", "Svenska Handelsbanken", "ASM International", "Adani Ports & SEZ", "ViacomCBS",
                "China Tower", "Shionogi", "Omron", "Entegris", "Thales", "Tech Mahindra", "J. B. Hunt", "Clorox",
                "Diamondback Energy", "Aviva", "Hindustan Zinc", "NVR", "China Railway Group", "Cooper Companies",
                "Toast", "Raymond James", "Fox Corporation", "AkzoNobel", "Symrise", "Citizens Financial Group",
                "Broadridge Financial Solutions", "Teledyne", "Hewlett Packard Enterprise", "Ventas", "UPM-Kymmene",
                "MGM Resorts", "RingCentral", "NetApp", "Krafton", "Caesars Entertainment", "Imperial Brands",
                "Qualtrics", "SS&C Technologies", "Robinhood", "Otsuka Holdings", "Suzuki Motor", "Avangrid",
                "Bath &amp", "Take 2 Interactive", "Quest Diagnostics", "M&T Bank", "Darden Restaurants",
                "Carlyle Group", "Cheniere Energy", "Signature Bank", "Domino's Pizza", "Burlington Stores",
                "Tata Steel", "Daiwa House", "KB Financial Group", "Wilmar", "Fifth Third Bank", "Royal Caribbean",
                "Nissan", "Snam", "Hologic", "Brown &amp", "Insulet", "Genuine Parts Company", "NICE", "Principal",
                "China Steel", "Splunk", ]
unicorn_list = ["Bytedance", "Stripe", "SpaceX", "Didi Chuxing", "Klarna", "Instacart", "Nubank", "Epic Games",
                "Databricks", "Rivian", "BYJU's", "One97 Communications", "Yuanfudao", "DJI Innovations", "Canva",
                "SHEIN", "Checkout.com", "Chime", "Grab", "Plaid Technologies", "Fanatics", "SenseTime", "JUUL Labs",
                "Manbang Group", "Bitmain Technologies", "Biosplice Therapeutics", "Robinhood", "Global Switch",
                "Celonis", "Aurora", "Lalamove", "Ripple", "OutSystems", "Klaviyo", "Roivant Sciences", "Northvolt",
                "Tanium", "Chehaoduo", "OYO Rooms", "goPuff", "ServiceTitan", "Tempus", "Xingsheng Selected", "Getir",
                "Caris Life Sciences", "J&T Express", "Brex", "Argo AI", "Gong", "We Doctor", "Discord", "Scale AI",
                "Faire", "Automation Anywhere", "Ziroom", "National Stock Exchange of India", "Compass", "Ola Cabs",
                "Samsara Networks", "Royole Corporation", "reddit", "GitLab", "Yuanqi Senlin", "Thrasio", "Better.com",
                "Easyhome", "Lianjia", "Airtable", "Vice Media", "Hopin", "Revolut", "Zomato", "Pony.ai",
                "Trade Republic", "Blockchain.com", "HashiCorp", "OneTrust", "Wise", "United Imaging Healthcare",
                "Hello TransTech", "Swiggy", "Krafton Game Union", "TripActions", "Nuro", "WM Motor", "Dream11", "Ro",
                "Howden Group Holdings", "SambaNova Systems", "Toast", "Magic Leap", "Snyk", "Meizu Technology",
                "Vinted", "Zenefits", "VIPKid", "Confluent", "UBTECH Robotics", "Outreach", "Bolt", "Ginkgo BioWorks",
                "Relativity Space", "Amplitude", "SSENSE", "Dataminr", "Houzz", "Yello Mobile", "MEGVII", "Niantic",
                "Greensill", "Impossible Foods", "Radiology Partners", "Next Insurance", "Patreon", "Kavak",
                "PointClickCare", "Zapier", "Clubhouse", "Benchling", "QuintoAndar", "Intarcia Therapeutics", "Gusto",
                "StockX", "MessageBird", "Devoted Health", "Guild Education", "Cohesity", "VAST Data", "Noom",
                "Relativity", "Otto Bock HealthCare", "Indigo Ag", "Bukalapak", "Freshworks", "Rappi", "N26",
                "HyalRoute", "Oxford Nanopore Technologies", "Youxia Motors", "Cloudwalk", "Rubrik",
                "Sila Nanotechnologies", "Blend", "Scopely", "Komodo Health", "WeRide", "Dadi Cinema", "Flexport",
                "Udemy", "Cedar", "Figure Technologies", "Back Market", "Huitongda", "Udaan", "HighRadius",
                "SentinelOne", "VANCL", "Warby Parker", "Yixia", "Xiaohongshu", "Traveloka", "SouChe Holdings",
                "BGL Group", "Circle", "Zuoyebang", "Talkdesk", "Netskope", "Horizon Robotics", "Wildlife Studios",
                "Razorpay", "Hinge Health", "Calendly", "BlockFi", "ActiveCampaign", "Pine Labs", "Forter", "Delhivery",
                "FlixMobility", "Kraken", "OpenAI", "Ovo", "Loft", "Workrise", "Bird Rides", "Meicai", "OakNorth",
                "Icertis", "DataRobot", "ContentSquare", "Weee!", "Graphcore", "Convoy", "MasterClass", "Sprinklr",
                "Tradeshift", "Toss", "Airwallex", "Dapper Labs", "Xinchao Media", "23andMe", "Vista Global", "BYTON",
                "Aihuishou", "Rapyd", "Cazoo", "Acronis", "AvidXchange", "Cgtz", "Star Charge", "Carbon", "Sportradar",
                "PolicyBazaar", "Duolingo", "Paxos", "Exabeam", "Collibra", "Bought By Many", "WEMAKEPRICE", "Uptake",
                "Skydance Media", "Lyra Health", "Greenlight", "Highspot", "Bowery Farming", "ReNew Power", "HEYTEA",
                "Zume", "Via", "Planet Labs", "NuCom Group", "Lyell Immunopharma", "Checkr", "MUSINSA", "Attentive",
                "CRED", "Current", "Bitso", "Huaqin Telecom Technology", "YITU Technology", "Cockroach Labs",
                "Nextdoor", "FirstCry", "C6 Bank", "Webflow", "6Sense", "Meesho", "ShareChat", "ReCharge", "Mambu",
                "Trendy Group International", "Avant", "Tubatu.com", "BlaBlaCar", "Quanergy Systems", "HuiMin", "Quora",
                "Improbable", "Kuaikan Manhua", "Preferred Networks", "LegalZoom", "4Paradigm", "Calm", "Kaseya",
                "Mafengwo", "Druva", "Babylon Health", "Kujiale", "AppsFlyer", "Notion Labs", "Figma",
                "Dingdong Maicai", "Keep", "Postman", "Redis Labs", "Unacademy", "Xingyun B2B", "Tipalti", "Unqork",
                "Chainalysis", "Virta Health", "ISN", "NAYUKI", "Earnix", "ATAI Life Sciences", "Clearco", "Hive",
                "KRY", "Urban Company", "Kajabi", "Ethos Technologies", "ThoughtSpot", "Formlabs", "Pipe",
                "Starling Bank", "InVision", "BillDesk", "eDaili", "monday.com", "Anduril", "MX Technologies",
                "Digit Insurance", "Aledade", "SpotOn", "ENOVATE", "Automattic", "ZocDoc", "Diamond Foundry", "AIWAYS",
                "Farmers Business Network", "GOAT", "Creditas", "Trulioo", "Apus Group", "BetterUp", "Buzzfeed",
                "Thumbtack", "Harry's Razor Company", "Allbirds", "PAX", "Carta", "VTEX", "Workato", "Harness",
                "Personio", "DispatchHealth", "Dutchie", "Wiz", "Unite Us", "CircleCI", "Alan", "XANT", "wefox",
                "Trader Interactive", "Jusfoun Big Data", "Zhubajie", "Monzo", "Infinidat", "Afiniti", "CAOCAO",
                "sweetgreen", "Seismic", "Verkada", "ASR Microelectronics", "Clari", "Ramp Financial", "Tonal", "Skims",
                "Clio", "SafetyCulture", "Extend", "ASAPP", "Cognite", "SmartHR", "Promasidor Holdings", "Illumio",
                "Baiwang", "Loom", "Ximalaya FM", "Mu Sigma", "ironSource", "TuJia", "Mofang Living", "Gett",
                "DT Dream", "Cybereason", "Changingedu", "XiaoZhu", "JOLLY Information Technology", "Yijiupi",
                "Cambridge Mobile Telematics", "Remitly", "Hippo Insurance", "Lenskart", "Podium", "ApplyBoard",
                "Mirakl", "Strava", "Olive", "Color", "Enfusion", "K Health", "Uplight", "ID.me", "Collective Health",
                "Handshake", "Snapdocs", "Chipper Cash", "Ledger", "Yipin Shengxian", "ECARX", "Bordrin Motors",
                "Coocaa", "Juma Peisong", "Ouyeel", "Gymshark", "Brii BioSciences", "Zeta", "Justworks", "Koudai",
                "Symphony", "Yidian Zixun", "Cabify", "Hive Box", "Deezer", "Away", "Voodoo", "Kong", "OwnBackup",
                "Epidemic Sound", "Yotpo", "Five Star Business Finance", "Phenom People", "GupShup", "Degreed",
                "Astranis Space Technologies", "ChargeBee Technologies", "Scalable Capital", "Veepee",
                "DeepBlue Technology", "Klook", "Rippling", "Plume", "Melio", "Signifyd", "Flipboard",
                "Grove Collaborative", "GPclub", "Tongdun Technology", "Alzheon", "Zeta Global", "Rocket Lab",
                "HeartFlow", "Sonder", "Trax", "You & Mr Jones", "InSightec", "Arctic Wolf Networks", "Everly Health",
                "Manner", "Innovaccer", "Socure", "Coalition", "ABL Space Systems", "Outschool", "Starry", "Intercom",
                "OVO Energy", "WTOIP", "ezCater", "KeepTruckin", "Applied Intuition", "BigID", "Rec Room", "Deel",
                "BrewDog", "EQRx", "GalaxySpace", "Judo Bank", "Yiguo", "Fair", "Glossier", "Zipline", "SmartNews",
                "Workhuman", "FiveTran", "Qumulo", "Dialpad", "Whoop", "Starburst", "Tealium", "Public", "Axonius",
                "Savage X Fenty", "Bitpanda", "Orca Security", "Pilot.com", "Paidy", "Cava Group", "Vectra Networks",
                "Ada Support", "Inari", "Project44", "Alation", "L&P Cosmetic", "Unisound", "Mininglamp Technology",
                "Sysdig", "Luoji Siwei", "Cerebras Systems", "Yimidida", "Modern Health", "Tuhu", "LifeMiles", "Venafi",
                "Doctolib", "Deposit Solutions", "TELD", "TangoMe", "AppDirect", "Juanpi", "OVH", "Eat Just",
                "GetYourGuide", "Ivalua", "Coveo", "Sisense", "Course Hero", "Sema4", "Pharmapacks", "SalesLoft",
                "Nexthink", "Zego", "Ginger", "Rightway", "Misfits Market", "Sunbit", "Biren Technology", "Nxin",
                "Rubicon Global", "Radius Payment Solutions", "Rivigo", "Dataiku", "Jiuxian", "Instabase", "Hesai Tech",
                "Proterra", "Sendbird", "Aprogen", "OrCam Technologies", "Leap Motor", ]
founder_list = []

stopwords = ["Therapeutics", "Finance", "energy", "worldwide", "global", "brands", "payments", "financial",
             "corporation", "Appliances", "investment", "trust", "Real Estate", "Games", "Communications", "Robotics",
             "Group", ".com", "Company", "Technologies", "health", "labs", "technology", "farming", "motor",
             "international", "automotive", "fashion", "solutions", "post", "figure", "citizens", "principal",
             "partners",
             "form", "current", "via", "american"]

unicorn_list = [x.lower() for x in unicorn_list]
company_list = [x.lower() for x in company_list]
stopwords = [x.lower() for x in stopwords]

for stop in stopwords:
    unicorn_list = [x.replace(stop, "").strip() for x in unicorn_list]
    company_list = [x.replace(stop, "").strip() for x in company_list]

ids = unicorn_list + company_list

# Number of words
query = "MATCH (r:edge) RETURN count(distinct([r.pos, r.run_index])) as nr_words, r.time as year"
res = semantic_network.db.receive_query(query)
nr_words = pd.DataFrame(res).sort_values(by="year")

query = "MATCH p=(a:word )-[:onto]->(r:edge) WHERE a.token in $idx Return a.token as token, count(distinct([r.pos, r.run_index])) as occ, r.time as year"
params = {}
params["idx"] = ids

res = semantic_network.db.receive_query(query, params)
df = pd.DataFrame(res)

df_total = df.groupby("token").sum()
df_total = df_total[df_total.occ > 50].sort_values(by=["token", "year"])
df_total["year"] = df_total["year"] / np.sum(range(1980, 2021))
firms_subsample = df_total[df_total["year"] < 1].index.to_list()

df = df[df.token.isin(firms_subsample)].sort_values(by=["token", "year"])

df = df.merge(nr_words, on="year", how="left").sort_values(by=["token", "year"])
df["norm_occ"] = 100000 * df.occ / df.nr_words
df_total = df.groupby("token").sum()
df_total["year"] = df_total["year"] / np.sum(range(1980, 2021))

firms = df_total.index.to_list()
empty_dict = {x: 0 for x in firms}
postcut = 0
allyear_list = []

for year in tqdm(times, desc="Iterating years", leave=False, colour='green', position=0):
    query = "MATCH p=(a:word)-[:onto]-(r:edge)-[:onto]-(b:word) WHERE b.token_id in $firms and a.token_id <> b.token_id and r.time in $times and r.weight >= " + str(
        postcut) + " return r.pos as position, r.run_index as ridx, collect(a.token_id) as alter, b.token_id as firm, sum(r.weight) as rweight, avg(r.sentiment) as sentiment, avg(r.subjectivity) as subjectivity"
    params = {}
    params["firms"] = semantic_network.ensure_ids(firms)
    params["times"] = [year]
    res = semantic_network.db.receive_query(query, params)
    rows = pd.DataFrame(res)
    rows=rows.sort_values(by=["ridx","position"])
    rows["firm_tk"] = semantic_network.ensure_tokens(rows["firm"])
    occurrences=rows.groupby(["ridx","position"]).sum().index.to_list()

    row_list = []
    for i, (ridx,pos) in tqdm(enumerate(occurrences), leave=False, desc="Iterating over occurrences in year {}".format(year),
                       position=1, colour='red', total=len(occurrences)):
        subdf_rows = rows[(rows.ridx == ridx) & (rows.position == pos)].copy()

        new_row = pd.Series(empty_dict).copy()
        new_row.loc["times"] = year
        new_row.loc["ridx"] = ridx
        new_row.loc["pos"] = pos
        new_row.loc["f_w_n"] = 0
        new_row.loc["f_w"] = 0
        new_row.loc["seq_length"] = 0
        # Add firm data
        for j,occrow in subdf_rows.iterrows():
            rweight = occrow["rweight"]
            new_row.loc[occrow.firm_tk] = rweight
            new_row.loc["subjectivity"] = occrow.subjectivity
            new_row.loc["sentiment"] = occrow.sentiment
            assert new_row[occrow.firm_tk] > 0
        # Add Founder data
        query = "Match (r:edge {pos:" + str(pos) + ", run_index:" + \
                str(ridx) + "})-[:seq]-(s:sequence)-[:seq]-(q:edge) WHERE q.pos<>r.pos " \
                        "WITH DISTINCT r,count(DISTINCT([q.pos,q.run_index])) as seq_length " \
                        "Match (r)-[:seq]-(s:sequence)-[:seq]-(q:edge) - [:onto]-(e:word) " \
                        "WHERE q.pos<>r.pos and e.token_id in $idx " \
                        "WITH DISTINCT q,r,e,seq_length " \
                        "RETURN r.pos as rpos, r.run_index as ridx," \
                        "sum(distinct(q.weight)) as cweight, e.token as context, " \
                        "seq_length, r.time as time order by context DESC"
        params = {}
        params["idx"] = semantic_network.ensure_ids(["founder"])
        res = pd.DataFrame(semantic_network.db.receive_query(query, params))
        if len(res)>0:
            res.loc[:, "cweight_n"] = res.loc[:, "cweight"] * 40 / res.loc[:, "seq_length"] * subdf_rows.rweight.sum()
            w = res.cweight.mean()
            wmin = res.cweight.min()
            wmax = res.cweight.max()
            w_n = res.cweight_n.mean()
            wmin_n = res.cweight_n.min()
            wmax_n = res.cweight_n.max()
            seq_length = res.seq_length.mean()
            new_row.loc["f_w_n"] = w_n * 100
            new_row.loc["f_w"] = w
            new_row.loc["seq_length"] = seq_length

        row_list.append(new_row)
    yeardf = pd.DataFrame(row_list)
    yeardf.to_excel(filename + "REGDF" + str(year) + ".xlsx", merge_cells=False)
    allyear_list.append(yeardf)
allyear_df = pd.concat(allyear_list)
allyear_df.to_excel(filename + "REGDF_allY_" + ".xlsx", merge_cells=False)
