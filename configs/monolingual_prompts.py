"""
Monolingual prompt translations for core evaluation instructions.

Used by the 'monolingual' strategy: the judge prompt is written in the
target language rather than English. This file provides the translated
instruction fragments for each supported language.

For languages not listed here, the pipeline falls back to English
(cross-lingual mode) and logs a warning.

IMPORTANT: These translations should be verified by native speakers
before final experiments. Mark any unverified translations with
verified=False.
"""

# Each entry: lang_code → dict of prompt fragments
# Keys: role, task_instruction, rating_scale, output_format
# These get interpolated into the main prompt template.

MONOLINGUAL_PROMPTS = {
    "fra": {
        "verified": True,
        "role": "Vous êtes un évaluateur expert en traitement du langage naturel.",
        "rate_instruction": "Évaluez la qualité de la réponse suivante sur une échelle de 1 à 5.",
        "scale": {
            1: "Très mauvais — complètement incorrect ou hors sujet",
            2: "Mauvais — erreurs majeures ou réponse très incomplète",
            3: "Acceptable — partiellement correct avec des erreurs notables",
            4: "Bon — majoritairement correct avec des erreurs mineures",
            5: "Excellent — correct, complet et bien formulé",
        },
        "output_format": "Donnez uniquement le score (1-5) suivi d'une brève justification.",
    },
    "spa": {
        "verified": True,
        "role": "Eres un evaluador experto en procesamiento de lenguaje natural.",
        "rate_instruction": "Evalúa la calidad de la siguiente respuesta en una escala de 1 a 5.",
        "scale": {
            1: "Muy malo — completamente incorrecto o fuera de tema",
            2: "Malo — errores graves o respuesta muy incompleta",
            3: "Aceptable — parcialmente correcto con errores notables",
            4: "Bueno — mayormente correcto con errores menores",
            5: "Excelente — correcto, completo y bien formulado",
        },
        "output_format": "Proporciona solo la puntuación (1-5) seguida de una breve justificación.",
    },
    "swa": {
        "verified": False,
        "role": "Wewe ni mtaalamu wa kutathmini katika usindikaji wa lugha asilia.",
        "rate_instruction": "Tathmini ubora wa jibu lifuatalo kwa kipimo cha 1 hadi 5.",
        "scale": {
            1: "Mbaya sana — si sahihi kabisa au nje ya mada",
            2: "Mbaya — makosa makubwa au jibu halijajitosheleza",
            3: "Wastani — sahihi kwa sehemu na makosa yanayoonekana",
            4: "Nzuri — sahihi kwa kiasi kikubwa na makosa madogo",
            5: "Bora — sahihi, kamili na limeandikwa vizuri",
        },
        "output_format": "Toa alama tu (1-5) ikifuatiwa na maelezo mafupi.",
    },
    "hau": {
        "verified": False,
        "role": "Kai gwani ne na kimanta harsunan kwamfuta.",
        "rate_instruction": "Ka tantance ingancin amsar da ke biye a kan ma'aunin 1 zuwa 5.",
        "scale": {
            1: "Mafi muni — ba daidai ba ko kuma ba shi da alaƙa",
            2: "Muni — akwai manyan kurakurai ko amsar ba ta cika ba",
            3: "Matsakaici — wani ɓangare daidai ne amma akwai kurakurai",
            4: "Mai kyau — mafi yawan sa daidai ne da ƙananan kurakurai",
            5: "Nagari — daidai, cikakke kuma an rubuta shi da kyau",
        },
        "output_format": "Ka ba da maki kawai (1-5) sannan ka ba da taƙaitaccen bayani.",
    },
    "yor": {
        "verified": False,
        "role": "Ìwọ jẹ́ amòye ayẹ̀wò nínú ìmọ̀-ẹ̀rọ èdè àdánidá.",
        "rate_instruction": "Ṣe àyẹ̀wò dídára ìdáhùn tó tẹ̀lé yìí lórí ìwọ̀n 1 sí 5.",
        "scale": {
            1: "Burú púpọ̀ — kò tọ́ rárá tàbí kò ní ìbátan",
            2: "Burú — àwọn àṣìṣe ńlá tàbí ìdáhùn tí kò pé",
            3: "Ó dára díẹ̀ — ó tọ́ ní apá kan pẹ̀lú àwọn àṣìṣe",
            4: "Ó dára — ó tọ́ ní ọ̀pọ̀ ibi pẹ̀lú àwọn àṣìṣe kékeré",
            5: "Ó dára jùlọ — ó tọ́, ó pé, ó sì kọ dáadáa",
        },
        "output_format": "Fun ìwọ̀n nìkan (1-5) tí a fi àlàyé kúkurú tẹ̀lé.",
    },
    "amh": {
        "verified": False,
        "role": "እርስዎ በተፈጥሮ ቋንቋ ሂደት ውስጥ የባለሙያ ገምጋሚ ነዎት።",
        "rate_instruction": "የሚከተለውን መልስ ጥራት ከ1 እስከ 5 ባለው ልኬት ይገምግሙ።",
        "scale": {
            1: "በጣም መጥፎ — ሙሉ ለሙሉ ትክክል ያልሆነ ወይም ከርዕሱ ውጪ",
            2: "መጥፎ — ከባድ ስህተቶች ወይም ያልተሟላ መልስ",
            3: "ተቀባይነት ያለው — በከፊል ትክክል ከሚታዩ ስህተቶች ጋር",
            4: "ጥሩ — በአብዛኛው ትክክል ከትንሽ ስህተቶች ጋር",
            5: "እጅግ በጣም ጥሩ — ትክክል፣ ሙሉ እና በጥሩ ሁኔታ የተቀመረ",
        },
        "output_format": "ውጤቱን (1-5) ብቻ ይስጡ ከዚያም አጭር ማብራሪያ ይጨምሩ።",
    },
    "cat": {
        "verified": True,
        "role": "Ets un avaluador expert en processament del llenguatge natural.",
        "rate_instruction": "Avalua la qualitat de la resposta següent en una escala de l'1 al 5.",
        "scale": {
            1: "Molt dolent — completament incorrecte o fora de tema",
            2: "Dolent — errors greus o resposta molt incompleta",
            3: "Acceptable — parcialment correcte amb errors notables",
            4: "Bo — majoritàriament correcte amb errors menors",
            5: "Excel·lent — correcte, complet i ben formulat",
        },
        "output_format": "Proporciona només la puntuació (1-5) seguida d'una breu justificació.",
    },
    "fas": {
        "verified": False,
        "role": "شما یک ارزیاب متخصص در پردازش زبان طبیعی هستید.",
        "rate_instruction": "کیفیت پاسخ زیر را در مقیاس ۱ تا ۵ ارزیابی کنید.",
        "scale": {
            1: "بسیار بد — کاملاً نادرست یا بی‌ربط",
            2: "بد — خطاهای جدی یا پاسخ بسیار ناقص",
            3: "قابل قبول — تا حدی درست با خطاهای قابل توجه",
            4: "خوب — عمدتاً درست با خطاهای جزئی",
            5: "عالی — درست، کامل و خوب نوشته شده",
        },
        "output_format": "فقط امتیاز (۱-۵) را بنویسید و سپس توضیح مختصری بدهید.",
    },
}

def get_monolingual_prompt(lang: str) -> dict | None:
    """Return monolingual prompt fragments for a language, or None if unavailable."""
    return MONOLINGUAL_PROMPTS.get(lang)
