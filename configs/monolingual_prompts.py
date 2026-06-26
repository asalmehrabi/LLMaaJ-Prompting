"""
Monolingual prompt translations for core evaluation instructions.

Used by the 'monolingual' strategy: the judge prompt is written in the
target language rather than English. This file provides the translated
instruction fragments for each supported language.

For languages not listed here, the pipeline falls back to English
(cross-lingual mode) and logs a warning.

Verified by native speakers: fra, spa, cat
Machine-translated (unverified): all others
"""

MONOLINGUAL_PROMPTS = {
    # ── VERIFIED ──────────────────────────────────────────────────────
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

    # ── MACHINE-TRANSLATED (unverified) ───────────────────────────────
    "ara": {
        "verified": False,
        "role": "أنت مقيّم خبير في معالجة اللغة الطبيعية.",
        "rate_instruction": "قيّم جودة الإجابة التالية على مقياس من 1 إلى 5.",
        "scale": {
            1: "سيء جداً — غير صحيح تماماً أو خارج الموضوع",
            2: "سيء — أخطاء جسيمة أو إجابة غير مكتملة",
            3: "مقبول — صحيح جزئياً مع أخطاء ملحوظة",
            4: "جيد — صحيح في معظمه مع أخطاء بسيطة",
            5: "ممتاز — صحيح وكامل ومُصاغ بشكل جيد",
        },
        "output_format": "أعطِ الدرجة فقط (1-5) ثم تبرير موجز.",
    },
    "deu": {
        "verified": False,
        "role": "Sie sind ein Expertengutachter für die Verarbeitung natürlicher Sprache.",
        "rate_instruction": "Bewerten Sie die Qualität der folgenden Antwort auf einer Skala von 1 bis 5.",
        "scale": {
            1: "Sehr schlecht — völlig falsch oder nicht relevant",
            2: "Schlecht — schwerwiegende Fehler oder sehr unvollständige Antwort",
            3: "Akzeptabel — teilweise korrekt mit merklichen Fehlern",
            4: "Gut — größtenteils korrekt mit kleineren Fehlern",
            5: "Ausgezeichnet — korrekt, vollständig und gut formuliert",
        },
        "output_format": "Geben Sie nur die Punktzahl (1-5) gefolgt von einer kurzen Begründung an.",
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
    "bem": {
        "verified": False,
        "role": "Muli ishike lya kusambilila umulimo wa kulankula amalweza ya natural.",
        "rate_instruction": "Chunguleni ubwino bwa ciyato ico ca kufwatila pa mipimo ya 1 ukufika 5.",
        "scale": {
            1: "Bubi sana — tachali cali ciweme ica kulandila",
            2: "Bubi — mingi ya makosa oo ciyato tachali cakwata",
            3: "Kwelela — cali ciweme mo cipande na makosa ya kumoneka",
            4: "Weme — cali ciweme bwingi na makosa acanono",
            5: "Ukusuma sana — cali ciweme, cakwata, no kulemba bwino",
        },
        "output_format": "Pele amaka (1-5) apafine na kufisulula panono.",
    },
    "ewe": {
        "verified": False,
        "role": "Ele wò nye aʋaƒoƒo ŋutinya ɖe nutome ƒe gɔmeɖeɖe dzidzime me.",
        "rate_instruction": "Kpɔ nuŋlɔɖiɖi si wò le edzi la ƒe nyuie ɖe mɔ si le 1 kple 5 dome.",
        "scale": {
            1: "Manyamanyanye — afɔsii nye o, alo mate ŋu ɖoa ŋu o",
            2: "Nyuie o — gadzikpɔkpɔ aʋadede kple nuŋlɔɖi si mevɛ o",
            3: "Nyo — nyo ɖe ʋa me kple gadzikpɔkpɔ siwo wogble",
            4: "Nyuie — nyuie ɖe ʋa me kple gadzikpɔkpɔ aɖewo",
            5: "Nyuie nyuie — aɖee, vɛ, eye woŋlɔ nyuie",
        },
        "output_format": "Na hafi (1-5) ʋo eye na ɖo nusɔsɔ aɖe.",
    },
    "fon": {
        "verified": False,
        "role": "Hwɛ́mɛ tɔn wɛ nyí wuntun tɔn ɖò xó gbigbɔ tɔn ɖaxó ɔ mɛ.",
        "rate_instruction": "Kpɔ́n alɔkpa e ɖò do bɔ é ko ɖ'ewu ɔ bo sɔ́ kɛnklɛn nú 1 jɛ 5.",
        "scale": {
            1: "Vɔ dó hú — é ɖò dó wɛ tawun alǒ é ɖó nǔ e é ɖó na ɖɔ ɔ gbɔn",
            2: "Vɔ — nǔ e é dó wɛ bo gbɛ́ é tɔn ɖó tɛnmɛ alǒ xósin e ko jɛ dó ɔ vɔ",
            3: "Alɔkpa ɖé — é nyí xósin e é ɖó na nyí ɔ ɖò ta ɖé kpo",
            4: "Mɔ̌ ɖé — é nyí tɛnmɛ tɔn ɖò nǔ gegě kpo nǔ kpɛví ɖé kpo",
            5: "Nyɔ́ — é nyí tɛnmɛ tɔn, é ko jɛ dó, bo dó wɛ nyi wɛ",
        },
        "output_format": "Ðó kɛnklɛn (1-5) géé bɔ à tlɛ nɔ xósin tɔn kpɛví.",
    },
    "ibo": {
        "verified": False,
        "role": "Ị bụ onye nchọcha nzọụkwụ na nhazi asụsụ.",
        "rate_instruction": "Nyochaa àgwà nzaghachi a dị n'okpuru n'ọnụ ọgụgụ 1 ruo 5.",
        "scale": {
            1: "Njọ nke ukwuu — ezighi ezi kari ọ bụ n'ime ihe ọzọ",
            2: "Njọ — ụjọ ndị ukwuu ma ọ bụ nzaghachi enweghị oke",
            3: "Ọ dị mma — ezighi ezi n'otu akụkụ na ụjọ ndị a na-ahụ anya",
            4: "Ọ dị mma — ezighi ezi n'ọtụtụ ihe na ụjọ obere",
            5: "Nke ọma — ezighi ezi, zuru oke, edere nke ọma",
        },
        "output_format": "Nye naanị ọnụ ọgụgụ (1-5) wee tinye nkọwa obere.",
    },
    "kin": {
        "verified": False,
        "role": "Uri impuguke yo gusuzuma mu gutunga ururimi rutunganye.",
        "rate_instruction": "Suzuma ireme ry'igisubizo gikurikira ku gipimo cya 1 kugeza 5.",
        "scale": {
            1: "Bibi cyane — si byo cyangwa ntibishoboka",
            2: "Bibi — amakosa manini cyangwa igisubizo kitarimo",
            3: "Byemewe — bihagije mu gice hamwe n'amakosa aboneka",
            4: "Byiza — bihagije cyane hamwe n'amakosa make",
            5: "Birasumba — bihagije, byuzuye kandi byanditswe neza",
        },
        "output_format": "Tanga gusa amanota (1-5) hanyuma usobanure gato.",
    },
    "lin": {
        "verified": False,
        "role": "Ozali linganga ya komeka na kobongisa maloba ya kozanga nzela.",
        "rate_instruction": "Tala bolamu ya eyano oyo ezali na nse na limite ya 1 tii 5.",
        "scale": {
            1: "Mabe mingi — ezali ndenge te te to ezali pembeni ya sujet",
            2: "Mabe — makosa ya monene to eyano ekoki te",
            3: "Ezali malamu — malamu na ndambo kasi na makosa oyo emonani",
            4: "Malamu — malamu mingi na makosa moke",
            5: "Kitoko — malamu, ekoki, mpe ekomami malamu",
        },
        "output_format": "Pesa kaka note (1-5) sima ya biyano moke.",
    },
    "lug": {
        "verified": False,
        "role": "Oli musomi omuzira mu kunoonyeza olulimi olw'obwangu.",
        "rate_instruction": "Gezako obulungi bw'okuddamu okuddamu okuri wansi ku kigero eky'okuva 1 okutuuka 5.",
        "scale": {
            1: "Bubi nnyo — si bya kwesiga oba si bya kwesiga",
            2: "Bubi — ensobi nnyingi oba okuddamu tekukwata",
            3: "Kigenderera — kyabulungi mu kitundu hamwe n'ensobi eziragika",
            4: "Bulungi — bulungi nnyo hamwe n'ensobi ntono",
            5: "Kirungi ennyo — kya kwesiga, kikwata, era kyawandiikibwa bulungi",
        },
        "output_format": "Wa amanota gokka (1-5) olwo owogerere obunene obuto.",
    },
    "orm": {
        "verified": False,
        "role": "Ati ogganaa ogeejjii qorannoo afaanii uumamaa keessatti.",
        "rate_instruction": "Gaarummaa deebii armaan gadii qabxii 1 hanga 5 irratti madaali.",
        "scale": {
            1: "Baay'ee gadhee — dhugaa miti yookiin mata duree hin qabu",
            2: "Gadhee — dogoggora gurguddaa yookiin deebiin guutuu miti",
            3: "Fudhatama qaba — tokko tokko sirriidha garuu dogoggorri mul'ata",
            4: "Gaarii — hedduu sirriidha dogoggora xiqqoon",
            5: "Baay'ee gaarii — sirrii, guutuu, akkaan barreeffame",
        },
        "output_format": "Qabxii qofaa (1-5) kenni itti aansees ibsa gabaabaa kenni.",
    },
    "pcm": {
        "verified": False,
        "role": "You be expert wey dey check natural language processing.",
        "rate_instruction": "Check how correct di answer wey dey below on scale of 1 to 5.",
        "scale": {
            1: "Very bad — totally wrong or e no relate",
            2: "Bad — big mistakes or di answer no complete",
            3: "Okay — partly correct but e get mistakes wey you go see",
            4: "Good — mostly correct with small mistakes",
            5: "Excellent — correct, complete and dem write am well",
        },
        "output_format": "Give only di score (1-5) then add small explanation.",
    },
    "som": {
        "verified": False,
        "role": "Adiga waxaad tahay khabiir ku takhasusay qiimaynta farsamada luuqadda dabiiciga ah.",
        "rate_instruction": "Qiimee tayada jawaabta hoose ee ku yaal miisaan 1 ilaa 5.",
        "scale": {
            1: "Aad u xun — gebi ahaantiis khalad ah ama aan xiriir lahayn",
            2: "Xun — khaladaad weyn ama jawaab aan dhamaystirnayn",
            3: "Aqbali karo — qayb ahaan sax ah oo khaladaad muuqda",
            4: "Wanaagsan — inta badan sax ah oo khaladaad yar",
            5: "Fiican — sax, dhamaystiran oo si fiican loo qoray",
        },
        "output_format": "Bixi oo kaliya dhibcaha (1-5) ka dib sharaxaad gaaban.",
    },
    "sot": {
        "verified": False,
        "role": "O setsebi sa ho lekola ho sebetsa ha puo ya tlhaho.",
        "rate_instruction": "Lekola boleng ba karabo e latelang hodima tekanyo ya 1 ho isa 5.",
        "scale": {
            1: "E mpe haholo — e fositse haholo kapa ha e amane le sehlooho",
            2: "E mpe — diphoso tse kgolo kapa karabo e sa phethahale",
            3: "E amohelehang — e nepahetse ka karolo le diphoso tse bonahalang",
            4: "E lokile — e nepahetse haholo le diphoso tse nyane",
            5: "E phethahetseng — e nepahetse, e phethehile, e ngotsweng hantle",
        },
        "output_format": "Fa meputso feela (1-5) ebe o tlhalosa hafofo.",
    },
    "tir": {
        "verified": False,
        "role": "ንስኻ ኣብ ናይ ቋንቋ ምስጢር ሓበሬታ ምፍናው ሞያዊ ኢኻ።",
        "rate_instruction": "ናይ ዝስዕብ መልሲ ጽፈት ካብ 1 ክሳብ 5 ዘሎ መዐቀኒ ግምት.",
        "scale": {
            1: "ብጣዕሚ ሕማቕ — ሙሉእ ብሙሉእ ጌጋ ወይ ናብ ርእሲ ጉዳይ ኣይኸደን",
            2: "ሕማቕ — ዓቢ ጌጋታት ወይ ምላሽ ኣይተዛዘመን",
            3: "ቅቡል — ኣካላዊ ቅኑዕ ምስ ዚረኤ ጌጋታት",
            4: "ጽቡቕ — ኣብ ብዙሕ ቅኑዕ ምስ ንኡሽቶ ጌጋታት",
            5: "ዝበለጸ — ቅኑዕ፣ ምሉእ፣ ብጽቡቕ ዝተጻሕፈ",
        },
        "output_format": "ነቲ ነጥቢ ጥራይ ሃብ (1-5) ቀጺሉ ሓጺር ምብርሃን ወስኽ።",
    },
    "twi": {
        "verified": False,
        "role": "Wo yɛ nimdeɛfo bi a ɔhwɛ asɛm a wɔde kasa abodwesɛm.",
        "rate_instruction": "Hwɛ mfatoho a ɛtọ so no ho papa wɔ ɔnkoronhwɛ a ɛfiri 1 kosi 5.",
        "scale": {
            1: "Bone paa — nni ho koraa anaa ɛnyɛ ho hwee",
            2: "Bone — mfomsoɔ kɛse anaasɛ mfatoho no nnyɛ a ɛho ato",
            3: "Yɛ kwan — ɛteetee faako bi ne mfomsoɔ a wohuu",
            4: "Yɛ — ɛteetee hunu mu ne mfomsoɔ nketewa",
            5: "Yɛ paa — teetee, ho atɔ, na wɔkyerew no yie",
        },
        "output_format": "Ma nkoronhwɛ (1-5) nko ara na fa nkyerɛkyerɛ ketewa ka ho.",
    },
    "wol": {
        "verified": False,
        "role": "Yëgël nga ci ci kanam yu rafet ci jëfandikukaay làkk bu dëkk.",
        "rate_instruction": "Xoolal njëgël bi ndaw bi ci suuf ci mbindukaayu 1 ak 5.",
        "scale": {
            1: "Dafa neex lool — dëgërul wala amul ci dëkkale",
            2: "Dafa neex — lolu bu mag rek wala tontu bi jëkkul",
            3: "Dafa yomb — jëm na ci kenn benn ak lolu yu gënë",
            4: "Dafa baax — baax na ci kanam bu bari ak lolu yu ndaw",
            5: "Dafa baax lool — dëgërul, mujj na, te bindal na ko yomb",
        },
        "output_format": "Jox solo (1-5) rekk, te jëfëndikool nettali bu topp.",
    },
    "xho": {
        "verified": False,
        "role": "Ungcali wokuhlola isiNgesi sesiNgesi sesiNgesi.",
        "rate_instruction": "Hlola umgangatho wempendulo elandelayo kwinqanaba eliphakathi kwe-1 ukuya kwi-5.",
        "scale": {
            1: "Mabi kakhulu — ayibi yihlo tu okanye ayinabudlelwane",
            2: "Mabi — iimpazamo ezinkulu okanye impendulo engaphelelanga",
            3: "Yamkelekile — ichanekile ngokuzondelela kunye neempazamo ezibonakalayo",
            4: "Ilungile — ichanekile kakhulu kunye neempazamo ezincinane",
            5: "Iphucukile — ichanekile, iphelele, kwaye ibhalwe kakuhle",
        },
        "output_format": "Nika kuphela amanqaku (1-5) emva koko wongeze ingcaciso emfutshane.",
    },
    "zul": {
        "verified": False,
        "role": "Ungumhlaziyi ochwepheshe wokuhluza ulimi lwemvelo.",
        "rate_instruction": "Hlola izinga lempendulo elandelayo kuma-1 kuya ku-5.",
        "scale": {
            1: "Kubi kakhulu — akukho nhlobo okulungile noma akuhlobene nesihloko",
            2: "Kubi — amaphutha amakhulu noma impendulo engaphelelanga",
            3: "Kuyamukeleka — kulungile ngokwengxenye kanye namaphutha abonakayo",
            4: "Kulungile — kulungile kakhulu kanye namaphutha amancane",
            5: "Kukuhle kakhulu — kulungile, kuphelele, kubhalwe kahle",
        },
        "output_format": "Nikeza amanani kuphela (1-5) bese ungeza incazelo emfishane.",
    },
}


def get_monolingual_prompt(lang: str) -> dict | None:
    """Return monolingual prompt fragments for a language, or None if unavailable."""
    return MONOLINGUAL_PROMPTS.get(lang)
