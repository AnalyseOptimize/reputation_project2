"""Явные правила, подобранные по значениям education/data/*.json.

Классифицируются поля программы/квалификации, а не название вуза.
Обновлять правила удобно по educationn/education_fields_audit.csv.
"""

FIELD_KEYS = (
    "field", "program", "program_or_specialty", "program_or_field",
    "field_or_program", "specialty", "specialization", "field_of_study", "major",
    "qualification", "degree", "degree_or_qualification", "credential",
)

LAW_PATTERN = (
    r"юриспруд|юридич|правовед|\bправ[оа]\b|правов|\bюрист\b|"
    r"\blaws?\b|\blegal\b|jurisprudence|\blawyer\b|\bll[mb]\b"
)
# Экономика понимается широко: финансы, учет, коммерция и бизнес-менеджмент.
# Техническое управление и государственное управление сами по себе не включены.
ECONOMICS_PATTERN = (
    r"эконом|финанс|бухгалтер|банков|аудит|налого|коммерц|внешн\w* торгов|"
    r"ценны\w* бумаг|менеджмент|маркетинг|делово\w* администр|бизнес|"
    r"\beconom\w*|\bfinanc\w*|\baccount\w*|\bbanking\b|\baudit\w*|"
    r"\bcommerce\b|\bbusiness\b|\bmarketing\b|\btaxation\b|"
    r"\b(?:e?mba|cfa|acca)\b|управлени\w* (?:компани|стоимост|организац)|"
    r"корпоративн\w* управлен|стратегическ\w* управлен|"
    r"(?:strategic|financial|general|business|corporate|project) management|"
    r"^management$|^управление развитием компании$"
)
MBA_PATTERN = r"\be\s*\.?m\s*\.?b\s*\.?a\b|\bm\s*\.?b\s*\.?a\b|мастер\w* делового администр|master of business administration"
RETRAINING_PATTERN = r"переподготов|переквалификац|professional retraining|vocational retraining|requalification"

# Положительное свидетельство зарубежного учреждения. Не считать зарубежным
# образование лишь потому, что название написано латиницей или содержит International.
FOREIGN_INSTITUTION_PATTERN = (
    r"harvard|stanford|insead|\bimd\b|international institute for management development|"
    r"oxford|cambridge|wharton|yale|princeton|kingston|london|manchester|"
    r"northumbria|durham|ashridge|henley|bocconi|rotterdam|erasmus|"
    r"berlin|freiburg|dortmund|hamburg|bremerhaven|\bwhu\b|"
    r"paris|escp|ecole|école|montpellier|sorbonne|geneva|switzerland|"
    r"stockholm|solvay|bruxelles|vienna|danish management|ie business|iese|"
    r"georgetown|columbia|dowling|duquesne|babson|emory|baruch|"
    r"case western|imperial college|loyola|michigan|murray state|missouri|"
    r"northeastern|northwestern|new york|rockefeller|st\.? john's|"
    r"california|arkansas|hawaii|maryland|oklahoma|rochester|texas|toledo|virginia|"
    r"united states naval|tuscarawas|washington and lee|mit sloan|mit academy|"
    r"bristol|hull|salford|the open university|chicago|cagliari|"
    r"hong kong|singapore|tel aviv|pretoria|beirut|baroda|"
    r"pontificia comillas|gdański|world islamic|"
    r"kazakh|kazakhstan|almaty|auezov|maqsut|taraz|american university of armenia|"
    r"kiev|riga|азербайджан|алма-ат|белорус|босфор|варшав|вебстер|вильнюс|"
    r"париж|берлин|ереван|казахск|караганд|киев|київ|кыргыз|ленинакан|"
    r"массачусет|минский|одесск|павлодар|приднестров|рижск|руднен|"
    r"ташкент|тбилис|семее|узбекск|валенси|витватерсранд|йоханнесбург|"
    r"хьюстон|боливара|усть-каменогор|нортумбр|лондон|италия|merton technical"
)
# Совместная программа в российском вузе не доказывает обучение за рубежом.
JOINT_DOMESTIC_PATTERN = r"ранхигс|мирбис|академии? народного хозяйства|институт бизнеса и экономики|корпоративный университет сбербанка|icef|cbsd"

# Алиасы относятся к юридическому лицу/историческому названию текущей компании.
# Дочерние общества не добавляются автоматически по общему бренду.
COMPANY_ALIASES = {
    "ABIO": ["Артген", "ИСКЧ", "Институт стволовых клеток человека"],
    "AFKS": ["АФК Система", "Sistema"],
    "AGRO": ["Русагро", "ГК Русагро", "ROS AGRO PLC", "Группа Русагро", "Группа Компаний Русагро"],
    "AQUA": ["ИНАРКТИКА", "Русская Аквакультура", "Inarctica"],
    "ASTR": ["Группа Астра"],
    "BELU": ["НоваБев Групп", "Novabev Group", "Beluga Group", "Синергия"],
    "CBOM": ["Московский кредитный банк", "МКБ", "Credit Bank of Moscow"],
    "CHMK": ["ЧМК", "Челябинский металлургический комбинат"],
    "DATA": ["Группа Аренадата", "Аренадата", "Arenadata"],
    "DELI": ["Каршеринг Руссия", "Делимобиль", "Delimobil"],
    "DVEC": ["ДЭК", "Дальневосточная энергетическая компания"],
    "ELFV": ["ЭЛ5-Энерго", "Энел Россия", "Энел ОГК-5", "ОГК-5", "Enel Russia", "EL5-Energo"],
    "ENPG": ["ЭН+ ГРУП", "En+ Group", "En+"],
    "ETLN": ["Etalon Group", "Группа Эталон"],
    "FEES": ["ФСК - Россети", "ФСК ЕЭС", "Федеральная сетевая компания - Россети"],
    "FESH": ["ДВМП", "Дальневосточное морское пароходство", "FESCO"],
    "FIVE": ["Корпоративный центр ИКС 5", "X5 Group", "X5 Retail Group", "X5 Retail Group N.V.", "Икс 5 Ритейл Груп Н.В."],
    "FIXP": ["Fix Price Group", "Fix Price", "Фикс Прайс Груп Лтд"],
    "GEMC": ["ЮМГ", "United Medical Group", "Европейский медицинский центр", "EMC"],
    "GLTR": ["Globaltrans Investment", "Globaltrans", "Глобалтранс Инвестмент ПЛС"],
    "GMKN": ["ГМК Норильский никель", "Норильский никель", "Норникель", "Norilsk Nickel"],
    "HEAD": ["Хэдхантер", "HeadHunter", "HeadHunter Group"],
    "HNFG": ["ЭЙЧ ЭФ ДЖИ", "ХЭНДЕРСОН ФЭШН ГРУПП", "Henderson Fashion Group"],
    "LEAS": ["ЛК Европлан", "Европлан", "Europlan"],
    "LSNGP": ["Россети Ленэнерго", "Ленэнерго"],
    "LSRG": ["Группа ЛСР", "LSR Group"],
    "MAGN": ["ММК", "Магнитогорский металлургический комбинат"],
    "MDMG": ["МД Медикал Груп", "MD Medical Group", "Мать и дитя"],
    "MGTSP": ["МГТС", "Московская городская телефонная сеть"],
    "MOEX": ["Московская Биржа", "Moscow Exchange"],
    "MRKC": ["Россети Центр", "МРСК Центра"],
    "MRKP": ["Россети Центр и Приволжье", "МРСК Центра и Приволжья"],
    "MRKU": ["Россети Урал", "МРСК Урала"],
    "MRKV": ["Россети Волга", "МРСК Волги"],
    "MRKZ": ["Россети Северо-Запад", "МРСК Северо-Запада"],
    "MSRS": ["Россети Московский регион", "МОЭСК"],
    "NLMK": ["НЛМК", "Новолипецкий металлургический комбинат", "Novolipetsk Steel", "NLMK Group"],
    "NMTP": ["НМТП", "Новороссийский морской торговый порт"],
    "OKEY": ["O'KEY Group", "О'КЕЙ ГРУПП", "О'Кей"],
    "OZON": ["Ozon Holdings", "Озон Холдингс ПиЭлСи", "Ozon"],
    "PIKK": ["ПИК-специализированный застройщик", "Группа ПИК", "ПИК"],
    "POLY": ["Solidcore Resources", "Polymetal International", "Полиметалл"],
    "POSI": ["Группа Позитив", "Positive Technologies"],
    "QIWI": ["QIWI", "NanduQ", "КИВИ ПиЭлСи"],
    "RKKE": ["РКК Энергия", "РКК Энергия им. С.П. Королева"],
    "RUAL": ["Объединенная компания РУСАЛ", "ОК РУСАЛ", "РУСАЛ", "United Company RUSAL"],
    "SBER": ["Сбербанк", "Сбербанк России", "Сбер", "Sberbank"],
    "SFIN": ["ЭсЭфАй", "SFI", "Сафмар Финансовые инвестиции"],
    "SGZH": ["ГК Сегежа", "Сегежа Групп", "Segezha Group"],
    "SMLT": ["ГК Самолет", "Группа Самолет", "Самолет"],
    "SVAV": ["СОЛЛЕРС", "Sollers"],
    "T": ["Т-Технологии", "ТКС Холдинг", "TCS Group Holding", "TCS Group"],
    "TATN": ["Татнефть", "Татнефть им. В.Д. Шашина", "Tatneft"],
    "TGKB": ["ТГК-2", "Территориальная генерирующая компания №2"],
    "TRMK": ["ТМК", "Трубная Металлургическая Компания", "TMK"],
    "UGLD": ["Южуралзолото Группа Компаний", "ЮГК", "Южуралзолото"],
    "UPRO": ["Юнипро", "Э.ОН Россия", "Э.ОН. Россия", "E.ON Russia", "Unipro"],
    "VKCO": ["ВК", "VK", "VK Company", "Mail.ru Group"],
    "VSEH": ["ВИ.ру", "ВсеИнструменты.ру"],
    "VSMO": ["Корпорация ВСМПО-АВИСМА", "ВСМПО-АВИСМА"],
    "VTBR": ["Банк ВТБ", "ВТБ", "VTB Bank"],
    "WUSH": ["ВУШ Холдинг", "Whoosh Holding"],
    "YDEX": ["Яндекс", "Yandex", "Yandex N.V."],
    "ZAYM": ["Микрофинансовая компания Займер", "МФК Займер", "Займер"],
}
