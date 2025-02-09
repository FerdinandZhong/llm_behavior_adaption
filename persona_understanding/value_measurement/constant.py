from os import path

PROFILE_KEYS = [
    "Gender",
    "Date of Birth",
    "Country",
    "Education Level",
    "Years of Experience",
    "Job Title",
]

PROFILE_TEMPLATE = """
{% for key, value in profile_data.items() %}
{{ key }}: {{ value }};
{% endfor %}
"""

LINE_BREAK = "*" * 50 + "\n"

OPTIONS_TEMPLATE = """
{% for option in option_list %}
{{ option }};
{% endfor %}
"""

DIRECT_VALUE_SELECTION_PROMPT = [
    {
        "role": "system",
        "content": "You are a chatbot designed to provide precise and personalized answers to questions based on the given user profile.  Analyze the question carefully and tailor your response to match the user's context.",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the user profile:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Below is the question:\n" + LINE_BREAK + "{question}\n" "{option_list}"
        ),
    },
]

CONVERSATION_HISTORY_PROMPT = [
    {
        "role": "system",
        "content": "You are a chatbot designed to provide precise and personalized answers to questions.  Analyze the question carefully and tailor your response to match the user's context.",
    },
    {
        "role": "user",
        "content": "Assume you are me, help me find the most sutiable answer to the following question.",
    },
    {
        "role": "user",
        "content": (
            "Below is the question:\n" + LINE_BREAK + "{question}\n" "{option_list}"
        ),
    },
]

format_str = """Answer the question in json format: 
{{
    \"option_id\": int, 
    \"reason\": str
}}"
"""

EXTRA_FORMAT = {
    "role": "user",
    "content": format_str,
}


main_dir = path.abspath(path.join(path.dirname(__file__), "..", ".."))

DIRECT_VALUE_QUESTIONS_CSV = path.join(
    main_dir,
    "datasets",
    "values_dataset",
    "vsm_direct_profile_questions_en_2024_Dec.csv",
)
DIALOGUE_CONTINUE_VALUE_QUESTIONS_CSV = path.join(
    main_dir,
    "datasets",
    "values_dataset",
    "vsm_dialogue_continue_questions_en_2024_Dec.csv",
)

DEFAULT_OPTION_IDS = [1, 2, 3, 4, 5]


COUNTRY_TO_CULTURE = {
    # Western (combined)
    "USA": "Western",
    "CAN": "Western",
    "GBR": "Western",
    "IRL": "Western",
    "AUS": "Western",
    "NZL": "Western",
    "FRA": "Western",
    "DEU": "Western",
    "ITA": "Western",
    "ESP": "Western",
    "PRT": "Western",
    "BEL": "Western",
    "NLD": "Western",
    "SWE": "Western",
    "NOR": "Western",
    "DNK": "Western",
    "FIN": "Western",
    "AUT": "Western",
    "CHE": "Western",
    # Latin American
    "ARG": "Latin American",
    "BRA": "Latin American",
    "CHL": "Latin American",
    "COL": "Latin American",
    "ECU": "Latin American",
    "PER": "Latin American",
    "VEN": "Latin American",
    "MEX": "Latin American",
    "URY": "Latin American",
    "PRY": "Latin American",
    "BOL": "Latin American",
    # Eastern Europe (merged Eastern and Central European)
    "ALB": "Eastern Europe",
    "BIH": "Eastern Europe",
    "BGR": "Eastern Europe",
    "HRV": "Eastern Europe",
    "CZE": "Eastern Europe",
    "SVK": "Eastern Europe",
    "SVN": "Eastern Europe",
    "HUN": "Eastern Europe",
    "MDA": "Eastern Europe",
    "MKD": "Eastern Europe",
    "ROU": "Eastern Europe",
    "RUS": "Eastern Europe",
    "SRB": "Eastern Europe",
    "UKR": "Eastern Europe",
    "GEO": "Eastern Europe",
    "EST": "Eastern Europe",
    "LVA": "Eastern Europe",
    "LTU": "Eastern Europe",
    # Middle Eastern / Middle Easternn & North Africa
    "IRN": "Middle Eastern",
    "IRQ": "Middle Eastern",
    "SAU": "Middle Eastern",
    "SYR": "Middle Eastern",
    "JOR": "Middle Eastern",
    "LBN": "Middle Eastern",
    "OMN": "Middle Eastern",
    "QAT": "Middle Eastern",
    "ARE": "Middle Eastern",
    "KWT": "Middle Eastern",
    "YEM": "Middle Eastern",
    "EGY": "Middle Eastern",
    "LBY": "Middle Eastern",
    # South Asian
    "IND": "South Asian",
    "NPL": "South Asian",
    "PAK": "South Asian",
    "BGD": "South Asian",
    "LKA": "South Asian",
    "MDV": "South Asian",
    # Eastern Asian
    "CHN": "Eastern Asian",
    "TWN": "Eastern Asian",
    "HKG": "Eastern Asian",
    "MAC": "Eastern Asian",
    "JPN": "Eastern Asian",
    "KOR": "Eastern Asian",
    "PRK": "Eastern Asian",
    "MNG": "Eastern Asian",
    # Southeast Asia
    "IDN": "Southeast Asian",
    "MYS": "Southeast Asian",
    "SGP": "Southeast Asian",
    "THA": "Southeast Asian",
    "PHL": "Southeast Asian",
    "VNM": "Southeast Asian",
    "KHM": "Southeast Asian",
    "LAO": "Southeast Asian",
    "MMR": "Southeast Asian",
    # Central Asia
    "KAZ": "Middle Eastern",
    "KGZ": "Middle Eastern",
    "TKM": "Middle Eastern",
    "UZB": "Middle Eastern",
    "TJK": "Middle Eastern",
    # Africa (broad grouping)
    "DZA": "African",
    "AGO": "African",
    "BEN": "African",
    "BWA": "African",
    "BFA": "African",
    "BDI": "African",
    "CMR": "African",
    "CPV": "African",
    "CAF": "African",
    "TCD": "African",
    "COM": "African",
    "COG": "African",
    "COD": "African",
    "CIV": "African",
    "DJI": "African",
    "GNQ": "African",
    "ERI": "African",
    "SWZ": "African",
    "ETH": "African",
    "GAB": "African",
    "GMB": "African",
    "GHA": "African",
    "GIN": "African",
    "GNB": "African",
    "KEN": "African",
    "LSO": "African",
    "LBR": "African",
    "MDG": "African",
    "MWI": "African",
    "MLI": "African",
    "MRT": "African",
    "MUS": "African",
    "MAR": "African",
    "MOZ": "African",
    "NAM": "African",
    "NER": "African",
    "NGA": "African",
    "RWA": "African",
    "STP": "African",
    "SEN": "African",
    "SLE": "African",
    "SOM": "African",
    "ZAF": "African",
    "SSD": "African",
    "SDN": "African",
    "TZA": "African",
    "TGO": "African",
    "TUN": "African",
    "UGA": "African",
    "ZMB": "African",
    "ZWE": "African",
    # Oceania
    "FJI": "Oceania",
    "PNG": "Oceania",
    "WSM": "Oceania",
    "VUT": "Oceania",
    "KIR": "Oceania",
    "MHL": "Oceania",
    "FSM": "Oceania",
    "NRU": "Oceania",
    "TON": "Oceania",
}


DEVELOPMENT_LEVEL = {
    # Developed Countries
    "USA": "Developed",
    "CAN": "Developed",
    "GBR": "Developed",
    "IRL": "Developed",
    "AUS": "Developed",
    "NZL": "Developed",
    "FRA": "Developed",
    "DEU": "Developed",
    "ITA": "Developed",
    "ESP": "Developed",
    "PRT": "Developed",
    "BEL": "Developed",
    "NLD": "Developed",
    "SWE": "Developed",
    "NOR": "Developed",
    "DNK": "Developed",
    "FIN": "Developed",
    "AUT": "Developed",
    "CHE": "Developed",
    "LUX": "Developed",
    "ISL": "Developed",
    "JPN": "Developed",
    "KOR": "Developed",
    "SGP": "Developed",
    # High‑income Middle Easternn states:
    "ARE": "Developed",
    "QAT": "Developed",
    "KWT": "Developed",
    "OMN": "Developed",
    "BHR": "Developed",
    # Developing Countries (middle-income)
    "CHN": "Developing",
    "IND": "Developing",
    "RUS": "Developing",  # Often considered transitional
    "MYS": "Developing",
    "THA": "Developing",
    "MEX": "Developing",
    "BRA": "Developing",
    "TUR": "Developing",
    "ZAF": "Developing",
    "IDN": "Developing",
    "VNM": "Developing",
    "PHL": "Developing",
    "COL": "Developing",
    "PER": "Developing",
    "ARG": "Developing",
    "PAK": "Developing",
    "BGD": "Developing",
    "NPL": "Developing",
    # Some Eastern European countries:
    "POL": "Developing",
    "HUN": "Developing",
    "CZE": "Developing",
    "SVK": "Developing",
    "EST": "Developing",
    "LVA": "Developing",
    "LTU": "Developing",
    # Central Asia (transitional economies)
    "KAZ": "Developing",
    "KGZ": "Developing",
    "TKM": "Developing",
    "UZB": "Developing",
    "TJK": "Developing",
    # Southeast Asia
    "KHM": "Developing",
    "LAO": "Developing",
    "MMR": "Developing",
    # Third World / Least Developed Countries (low-income)
    "AFG": "Third World",
    "SOM": "Third World",
    "NER": "Third World",
    "TCD": "Third World",
    "SDN": "Third World",
    "BFA": "Third World",
    "MLI": "Third World",
    "BDI": "Third World",
    "RWA": "Third World",
    "SSD": "Third World",
    "ZWE": "Third World",
    "ZMB": "Third World",
    "AGO": "Third World",
    "CAF": "Third World",
    "COG": "Third World",
    "GNQ": "Third World",
    "ERI": "Third World",
    "LBR": "Third World",
    "MDG": "Third World",
    "MRT": "Third World",
    "SLE": "Third World",
    "SWZ": "Third World",
    "TGO": "Third World",
    # Additional African countries:
    "ETH": "Third World",
    "UGA": "Third World",
    "KEN": "Third World",
    "MOZ": "Third World",
    # Oceania (most are small islands; classification can vary)
    "FJI": "Developing",
    "PNG": "Developing",
    "WSM": "Developing",
    "VUT": "Developing",
    "KIR": "Developing",
    "MHL": "Developing",
    "FSM": "Developing",
    "NRU": "Developing",
    "TON": "Developing",
    # Europe additional (some European countries not classified above are generally developed)
    "SCO": "Developed",  # Scotland (if using separate codes)
    "NIR": "Developed",  # Northern Ireland
    # Note: Most European Union countries have very high HDI,
    # so you might want to classify them as Developed.
    # (This mapping does not cover every single country.
    # For a full mapping, you may wish to integrate data from the UNDP or World Bank.)
}
