"""Language identification for transcripts — no model download, no network.

The ASR model is multilingual but says nothing about *which* language it just
heard. Two sources of evidence are combined here:

  * **script** — a hard signal. Greek, Cyrillic, Arabic, Hebrew, Hangul, kana,
    Devanagari and Thai cannot be confused with a Latin-script language, and for
    those the answer is available from one page of text.
  * **function words** — for Latin-script languages the closed-class vocabulary
    (articles, pronouns, prepositions, auxiliaries) is both the most frequent
    part of speech and the most language-specific. Content words are shared
    across languages ("migration", "protocol") and are deliberately ignored.

Diacritics and a handful of characteristic clusters add weight where a text is
too short for function words to separate two close languages (Norwegian/Danish,
Spanish/Portuguese, Czech/Slovak).

`detect()` returns a code, a human name, a confidence and the runner-up scores
so a caller can decide to show "English?" instead of asserting a wrong answer.
`rollup()` aggregates many per-window results into a document-level verdict and
flags genuinely bilingual recordings instead of picking a winner silently.
"""

from __future__ import annotations

import re
from collections import Counter

# code -> (English name, native name, right-to-left)
LANGUAGES: dict[str, tuple[str, str, bool]] = {
    "en": ("English", "English", False),
    "de": ("German", "Deutsch", False),
    "fr": ("French", "Français", False),
    "es": ("Spanish", "Español", False),
    "it": ("Italian", "Italiano", False),
    "pt": ("Portuguese", "Português", False),
    "nl": ("Dutch", "Nederlands", False),
    "sv": ("Swedish", "Svenska", False),
    "da": ("Danish", "Dansk", False),
    "nb": ("Norwegian", "Norsk", False),
    "fi": ("Finnish", "Suomi", False),
    "pl": ("Polish", "Polski", False),
    "cs": ("Czech", "Čeština", False),
    "sk": ("Slovak", "Slovenčina", False),
    "hu": ("Hungarian", "Magyar", False),
    "ro": ("Romanian", "Română", False),
    "tr": ("Turkish", "Türkçe", False),
    "id": ("Indonesian", "Bahasa Indonesia", False),
    "hr": ("Croatian", "Hrvatski", False),
    "sl": ("Slovenian", "Slovenščina", False),
    "ru": ("Russian", "Русский", False),
    "uk": ("Ukrainian", "Українська", False),
    "bg": ("Bulgarian", "Български", False),
    "sr": ("Serbian", "Српски", False),
    "el": ("Greek", "Ελληνικά", False),
    "ar": ("Arabic", "العربية", True),
    "fa": ("Persian", "فارسی", True),
    "he": ("Hebrew", "עברית", True),
    "hi": ("Hindi", "हिन्दी", False),
    "th": ("Thai", "ไทย", False),
    "zh": ("Chinese", "中文", False),
    "ja": ("Japanese", "日本語", False),
    "ko": ("Korean", "한국어", False),
}

LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "nb", "fi",
         "pl", "cs", "sk", "hu", "ro", "tr", "id", "hr", "sl"}
CYRILLIC = {"ru", "uk", "bg", "sr"}

# Closed-class vocabulary per language. Weighted by how often the word really
# occurs in speech: 2 for articles/pronouns/conjunctions, 1 for the rest.
_SW: dict[str, dict[str, int]] = {
    "en": dict.fromkeys(
        "the a an and or but if then than that this these those of to in on at by for with from "
        "as is are was were be been being am do does did done have has had not no yes i you he she "
        "it we they me him her us them my your his its our their there here what when where who why "
        "how all some any more most other such only just also very can could will would shall should "
        "may might must about into over under after before between during because while so up out off "
        "again still already really actually think know want need going going get got make made".split(), 1),
    "de": dict.fromkeys(
        "der die das den dem des ein eine einen einem einer eines und oder aber wenn dann als dass "
        "ich du er sie es wir ihr mich dich ihn uns euch mir dir ihm ihnen mein dein sein unser euer "
        "ist sind war waren bin bist sein gewesen habe hast hat haben hatte hatten wird werden wurde "
        "worden nicht kein keine keinen mit von zu zum zur im in am an auf für aus bei nach über vor "
        "durch gegen ohne um unter zwischen auch noch schon nur sehr wie was wer wo wann warum weil "
        "damit dass man sich doch mal jetzt hier dort also aber bitte danke ja nein gut schon kann "
        "können muss müssen soll sollen will wollen mag möchte machen macht gemacht gehen geht".split(), 1),
    "fr": dict.fromkeys(
        "le la les un une des du de d et ou mais si alors que qui quoi dont où quand comme je tu il "
        "elle on nous vous ils elles me te se lui leur mon ton son notre votre leurs ce cet cette ces "
        "est sont était étaient être avoir ai as a avons avez ont avait avaient ne pas plus moins très "
        "dans sur sous avec sans pour par chez vers entre aussi encore déjà bien tout tous toute toutes "
        "oui non merci bonjour voilà donc car parce peut peuvent doit dois faut fait faire dit dire".split(), 1),
    "es": dict.fromkeys(
        "el la los las un una unos unas de del al a y o pero si entonces que quien quienes como donde "
        "cuando porque yo tú él ella usted nosotros vosotros ellos ellas me te se nos os le les mi tu "
        "su nuestro vuestro este esta estos estas ese esa eso es son era eran ser estar está están hay "
        "ha han había no sí más menos muy ya también todavía bien todo todos toda todas por para con sin "
        "sobre entre desde hasta cuando ahora aquí allí gracias hola bueno vale pues puede pueden debe".split(), 1),
    "it": dict.fromkeys(
        "il lo la i gli le un uno una di del della dei delle al alla allo agli alle e ed o ma se che "
        "chi cui come dove quando perché io tu lui lei noi voi loro mi ti ci vi si ne è sono era erano "
        "essere avere ho hai ha abbiamo avete hanno non no sì più meno molto già anche ancora bene tutto "
        "tutti tutta tutte per con senza su sopra sotto tra fra da in a nel nella nelle negli allo dopo "
        "prima adesso qui qua grazie ciao buono va può possono deve devono fatto fare detto dire".split(), 1),
    "pt": dict.fromkeys(
        "o a os as um uma uns umas de do da dos das ao à aos às e ou mas se que quem como onde quando "
        "porque eu tu ele ela você nós eles elas me te se nos vos lhe lhes meu teu seu nosso vosso este "
        "esta estes estas esse essa isso é são era eram ser estar está estão há tem têm havia não sim "
        "mais menos muito já também ainda bem todo todos toda todas por para com sem sobre entre desde "
        "até agora aqui ali obrigado obrigada olá bom boa pode podem deve devem fazer feito dizer dito".split(), 1),
    "nl": dict.fromkeys(
        "de het een en of maar als dan dat die deze dit ik jij hij zij wij jullie ze mij jou hem haar ons "
        "hen hun mijn jouw zijn haar onze jullie hun is zijn was waren ben bent wezen hebben heeft heb "
        "hebt had hadden niet geen met van voor naar bij op in aan uit over onder tussen door tegen "
        "zonder om ook nog al alleen zeer hoe wat wie waar wanneer waarom omdat dus toch nu hier daar "
        "goed ja nee dank hallo kan kunnen moet moeten zal zullen wil willen doen doet gedaan gaan gaat".split(), 1),
    "sv": dict.fromkeys(
        "och i att det som en ett på är av för med till den har de inte om ett jag du han hon vi ni dom "
        "mig dig honom henne oss er min mitt mina din ditt hans hennes vår vårt våra denna detta dessa "
        "var vara varit blir blev kan kunna ska skulle vill vilja måste men eller så också bara mycket "
        "här där nu då när varför eftersom tack hej ja nej bra gör göra gjorde säger säga sa".split(), 1),
    "da": dict.fromkeys(
        "og i at det som en et på er af for med til den har de ikke om jeg du han hun vi I dem mig dig "
        "ham hende os jer min mit mine din dit hans hendes vores deres denne dette disse var være været "
        "bliver blev kan kunne skal skulle vil ville må men eller så også kun meget her der nu da når "
        "hvorfor fordi tak hej ja nej god gør gøre gjorde siger sige sagde".split(), 1),
    "nb": dict.fromkeys(
        "og i at det som en et på er av for med til den har de ikke om jeg du han hun vi dere dem meg deg "
        "ham henne oss deres min mitt mine din ditt hans hennes vår våre denne dette disse var være vært "
        "blir ble kan kunne skal skulle vil ville må men eller så også bare mye her der nå da når hvorfor "
        "fordi takk hei ja nei bra gjør gjøre gjorde sier si sa".split(), 1),
    "fi": dict.fromkeys(
        "ja on ei se että hän me te he minä sinä minun sinun meidän teidän heidän tämä tuo nämä ne oli "
        "ovat olen olet olemme olette ollut ollut ei en et emme ette eivät mutta tai jos kun niin myös "
        "vain paljon nyt sitten missä milloin miksi koska kiitos hei kyllä hyvä tehdä tekee teki sanoa "
        "sanoo sanoi mukaan kanssa ilman yli alla".split(), 1),
    "pl": dict.fromkeys(
        "i w na z do nie że się to jest są był była było były być mam masz ma mamy macie mają miał nie "
        "tak jak co kto gdzie kiedy dlaczego bo ale lub czy już jeszcze tylko bardzo też tu tam teraz "
        "ten ta te tego tej tym tych mój twoja jego jej nasz wasz ich dla od po pod nad przy przez bez "
        "między o u dzięki proszę dziękuję dzień dobry dobrze może mogą musi muszą robić zrobić mówić".split(), 1),
    "cs": dict.fromkeys(
        "a v na z do že se to je jsou byl byla bylo byly být mám máš má máme máte mají měl ne ano jak co "
        "kdo kde kdy proč protože ale nebo jestli už ještě jen velmi také tady tam teď ten ta to toho té "
        "tím těch můj tvoje jeho její náš váš jejich pro od po pod nad při přes bez mezi o u díky prosím "
        "děkuji dobrý den dobře může mohou musí muset dělat udělat říkat".split(), 1),
    "sk": dict.fromkeys(
        "a v na z do že sa to je sú bol bola bolo boli byť mám máš má máme máte majú mal nie áno ako čo "
        "kto kde kedy prečo lebo ale alebo či už ešte len veľmi tiež tu tam teraz ten tá to toho tej tým "
        "tých môj tvoje jeho jej náš váš ich pre od po pod nad pri cez bez medzi o u vďaka prosím ďakujem "
        "dobrý deň dobre môže môžu musí musieť robiť urobiť hovoriť".split(), 1),
    "hu": dict.fromkeys(
        "a az és vagy de hogy nem is egy van volt lesz vagyok vagyunk vagytok vannak nekem neked neki "
        "nekünk nektek nekik én te ő mi ti ők engem téged őt minket titeket őket ez azt ezt ezek azok "
        "itt ott most akkor miért mert még már csak nagyon is jól köszönöm szia igen jó csinál csinálni "
        "mond mondani volt kell lehet".split(), 1),
    "ro": dict.fromkeys(
        "și în de la cu pe din un o care că nu se este sunt era erau fi fost am ai are avem aveți au "
        "eu tu el ea noi voi ei ele mă te îl ne vă îi acest această acești aceste asta asta ce cine unde "
        "când de ce pentru că dar sau dacă deja încă doar foarte aici acolo acum bine mulțumesc salut da "
        "poate pot trebuie face făcut spune zis".split(), 1),
    "tr": dict.fromkeys(
        "ve bir bu şu o da de ne ki için ile olan olarak var yok değil çok daha en gibi ama veya eğer "
        "ben sen biz siz onlar benim senin onun bizim sizin onların bana sana ona bize size onlara beni "
        "seni onu ne nasıl neden nerede ne zaman çünkü şimdi burada orada iyi evet hayır teşekkür merhaba "
        "yapmak yapar yaptı söylemek söyler dedi olmak oldu olur".split(), 1),
    "id": dict.fromkeys(
        "dan yang di ke dari untuk dengan pada ini itu tidak ada saya kamu dia kami kita mereka aku anda "
        "nya adalah akan sudah belum bisa harus tetapi tapi atau jika karena juga sangat hanya masih "
        "saja lagi sekarang di sini sana apa siapa bagaimana mengapa kapan baik ya terima kasih halo "
        "melakukan membuat bilang kata".split(), 1),
    "hr": dict.fromkeys(
        "i u na za od do da se je su bio bila bilo bili biti imam imaš ima imamo imate imaju nije ne da "
        "kako što tko gdje kada zašto jer ali ili već još samo vrlo također ovdje tamo sada ovaj ova ovo "
        "tog te taj moj tvoj njegov njezin naš vaš njihov za s prema bez između o kod hvala molim dobar "
        "dan dobro može mogu mora moraju raditi napraviti reći".split(), 1),
    "sl": dict.fromkeys(
        "in v na za od do da se je so bil bila bilo bili biti imam imaš ima imamo imate imajo ni ne da "
        "kako kaj kdo kje kdaj zakaj ker ali ali že še samo zelo tudi tukaj tam zdaj ta to tega te tisti "
        "moj tvoj njegov njen naš vaš njihov za s proti brez med o pri hvala prosim dober dan dobro lahko "
        "morajo delati narediti reči".split(), 1),
    "ru": dict.fromkeys(
        "и в не на что он она оно мы вы они я ты это как так же но а или если то что чтобы был была было "
        "были быть есть был будет буду будешь будем будете будут меня тебя его её нас вас их мне тебе ему "
        "ей нам вам им мой твой его её наш ваш их этот эта это эти тот та то те здесь там теперь когда "
        "почему потому что уже ещё только очень тоже можно нужно надо делать сделать говорить сказать".split(), 1),
    "uk": dict.fromkeys(
        "і в не на що він вона воно ми ви вони я ти це як так же але або якщо то що щоб був була було були "
        "бути є буде буду будеш будемо будете будуть мене тебе його її нас вас їх мені тобі йому їй нам "
        "вам їм мій твій його її наш ваш їх цей ця це ці той та те ті тут там тепер коли чому тому що вже "
        "ще тільки дуже теж можна треба робити зробити говорити сказати".split(), 1),
    "bg": dict.fromkeys(
        "и в не на че той тя то ние вие те аз ти това как така също но или ако то че да беше бил била било "
        "били бъда съм си сме сте са ме те го я ни ви ги ми ти му ѝ ние вие им мой твой негов неин наш ваш "
        "техен този тази това тези онзи тук там сега кога защо защото вече още само много също може трябва "
        "прави направи казва каже".split(), 1),
    "sr": dict.fromkeys(
        "и у на за од до да се је су био била било били бити имам имаш има имамо имате имају није не да "
        "како шта ко где када зашто јер али или већ још само веома такође овде тамо сада овај ова ово "
        "тог те тај мој твој његов њен наш ваш њихов за са према без између о код хвала молим добар дан "
        "добро може могу мора морају радити направити рећи".split(), 1),
}

# Diacritics that strongly imply one language (or a small group).
_DIACRITICS: dict[str, str] = {
    "de": "äöüß",
    "fr": "àâæçéèêëîïôœùûüÿ",
    "es": "áéíóúñ¿¡ü",
    "pt": "áâãàçéêíóôõú",
    "it": "àèéìíòóùú",
    "nl": "ëïéè",
    "sv": "åäö",
    "da": "æøå",
    "nb": "æøå",
    "fi": "äö",
    "pl": "ąćęłńóśźż",
    "cs": "áčďéěíňóřšťúůýž",
    "sk": "áäčďéíĺľňóôŕšťúýž",
    "hu": "áéíóöőúüű",
    "ro": "ăâîșțşţ",
    "tr": "çğıöşü",
    "hr": "čćđšž",
    "sl": "čšž",
}

# Characteristic clusters — weak evidence, only breaks near-ties.
_CLUSTERS: dict[str, tuple[str, ...]] = {
    "en": ("th", "wh", "sh", "ing", "ght"),
    "de": ("sch", "ung", "ich", "eit", "ch"),
    "fr": ("que", "ent", "ion", "ais", "ez"),
    "es": ("ción", "que", "ado", "ido", "os "),
    "it": ("zione", "gli", "ato", "ito", "che"),
    "pt": ("ção", "nh", "lha", "eiro", "ões"),
    "nl": ("ij", "sch", "aan", "lijk", "oor"),
    "sv": ("och", "att", "ing", "het", "ska"),
    "da": ("og", "ikke", "ing", "else", "skab"),
    "nb": ("og", "ikke", "ing", "else", "skap"),
    "pl": ("sz", "cz", "rz", "prz", "nie"),
    "cs": ("př", "stř", " ou", "nost", "ční"),
    "tr": ("lar", "ler", "yor", "mek", "mak"),
}

# Scripts that can be decided from the alphabet alone.
_SCRIPT_RANGES = [
    ("Latin", r"A-Za-z\u00c0-\u024f\u1e00-\u1eff"),
    ("Cyrillic", r"\u0400-\u04ff"),
    ("Greek", r"\u0370-\u03ff\u1f00-\u1fff"),
    ("Arabic", r"\u0600-\u06ff\u0750-\u077f"),
    ("Hebrew", r"\u0590-\u05ff"),
    ("Devanagari", r"\u0900-\u097f"),
    ("Thai", r"\u0e00-\u0e7f"),
    ("Hangul", r"\uac00-\ud7af\u1100-\u11ff"),
    ("Kana", r"\u3040-\u30ff"),
    ("Han", r"\u4e00-\u9fff\u3400-\u4dbf"),
]

_SCRIPT_RES = {name: re.compile(f"[{rng}]") for name, rng in _SCRIPT_RANGES}
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

_MIN_WORDS = 3


def _count_scripts(text: str) -> Counter:
    counts = Counter()
    for name, rx in _SCRIPT_RES.items():
        n = len(rx.findall(text))
        if n:
            counts[name] = n
    return counts


def _cyrillic_language(text: str) -> tuple[str, float]:
    low = text.lower()
    words = _TOKEN_RE.findall(low)
    # Script-level giveaways first: these letters exist in one language only.
    if any(ch in low for ch in "їєґ"):
        return "uk", 0.93
    if any(ch in low for ch in "ђјљњћџ"):
        return "sr", 0.93
    if any(ch in low for ch in "ѓќѕ"):
        return "sr", 0.85
    scores = {code: 0 for code in CYRILLIC}
    for w in words:
        for code in CYRILLIC:
            if w in _SW[code]:
                scores[code] += 1
    total = sum(scores.values())
    if total:
        order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top, second = order[0][1], order[1][1]
        conf = 0.55 + 0.4 * (top - second) / max(top, 1)
        return order[0][0], round(min(0.97, conf), 3)
    # No evidence: ъ + ы/э is Russian, ъ with none of them is ambiguous but
    # Bulgarian is the safest guess for the ъ-heavy case.
    if "ы" in low or "э" in low or "ё" in low:
        return "ru", 0.6
    if "ъ" in low:
        return "bg", 0.5
    return "ru", 0.4


def _latin_scores(text: str) -> tuple[dict[str, float], int]:
    low = text.lower()
    words = _TOKEN_RE.findall(low)
    scores = {code: 0.0 for code in LATIN}
    for w in words:
        for code in LATIN:
            weight = _SW[code].get(w)
            if weight:
                scores[code] += weight
    # Diacritics: one occurrence is already strong evidence, but cap the bonus
    # so a single "é" cannot outvote a page of function words.
    for code, chars in _DIACRITICS.items():
        hits = sum(low.count(ch) for ch in chars)
        if hits:
            scores[code] += min(4.0, 0.8 * hits)
    for code, clusters in _CLUSTERS.items():
        hits = sum(min(2, low.count(c)) for c in clusters if c in low)
        if hits:
            scores[code] += 0.35 * hits
    return scores, len(words)


def _pick(scores: dict[str, float], words: int) -> tuple[str, float, list[dict], bool, str | None]:
    ranked = sorted(((c, s) for c, s in scores.items() if s > 0), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "", 0.0, [], False, None
    top_code, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = (top_score - second) / max(top_score, 1e-9)
    evidence = top_score
    # Confidence needs both a clear win and enough material to have won on.
    # Three words can never be more than a hint, however clean the margin.
    length_factor = min(1.0, words / 12.0) * 0.6 + 0.4 * min(1.0, words / 4.0)
    conf = (0.35 + 0.65 * margin) * length_factor if top_score >= 2 else 0.0
    alts = [{"code": c, "name": LANGUAGES.get(c, (c, c, False))[0],
             "score": round(s / max(evidence, 1e-9), 3)} for c, s in ranked[1:4]]
    # Code-switching is real in Austrian offices: a window can hold two
    # languages and the UI must be allowed to say so instead of asserting one.
    mixed = bool(second / max(evidence, 1e-9) >= 0.35 and top_score >= 4)
    secondary = ranked[1][0] if mixed else None
    return top_code, round(max(0.0, min(0.99, conf)), 3), alts, mixed, secondary


def detect(text: str) -> dict:
    """Identify the language of one piece of text.

    Returns ``{"code", "name", "native", "rtl", "confidence", "script",
    "words", "alternatives", "mixed", "secondary"}``. ``code`` is ``""`` when
    there is not enough signal — callers should render that as unknown rather
    than guessing.
    """
    text = (text or "").strip()
    words = _TOKEN_RE.findall(text.lower())
    empty = {"code": "", "name": "Unknown", "native": "", "rtl": False,
             "confidence": 0.0, "script": "", "words": len(words),
             "alternatives": [], "mixed": False, "secondary": None}
    if len(words) < _MIN_WORDS:
        return empty

    scripts = _count_scripts(text)
    if not scripts:
        return empty
    dominant, count = scripts.most_common(1)[0]
    total_letters = sum(scripts.values())
    share = count / max(total_letters, 1)

    code, conf, alts = "", 0.0, []
    mixed, secondary = False, None
    if dominant in ("Kana", "Han", "Hangul", "Greek", "Hebrew", "Devanagari", "Thai", "Arabic"):
        if dominant == "Kana":
            code, conf = "ja", 0.95
        elif dominant == "Han":
            code, conf = "zh", 0.9 * share
        elif dominant == "Hangul":
            code, conf = "ko", 0.95
        elif dominant == "Greek":
            code, conf = "el", 0.95
        elif dominant == "Hebrew":
            code, conf = "he", 0.95
        elif dominant == "Devanagari":
            code, conf = "hi", 0.95
        elif dominant == "Thai":
            code, conf = "th", 0.95
        else:  # Arabic script: Persian is the only close neighbour worth testing
            low = text.lower()
            persian = sum(1 for ch in "پچژگکی" if ch in low)
            code, conf = ("fa", 0.9) if persian >= 2 else ("ar", 0.85)
    elif dominant == "Cyrillic":
        code, conf = _cyrillic_language(text)
    else:
        scores, n = _latin_scores(text)
        code, conf, alts, mixed, secondary = _pick(scores, n)
        if not code:
            return empty
        conf *= share

    if not code:
        return empty
    name, native, rtl = LANGUAGES.get(code, (code.upper(), code, False))
    return {
        "code": code, "name": name, "native": native, "rtl": rtl,
        "confidence": round(conf, 3), "script": dominant, "words": len(words),
        "alternatives": alts, "mixed": mixed, "secondary": secondary,
    }


def merge(results: list[dict]) -> dict:
    """Aggregate per-window detections into one document-level verdict.

    Windows are weighted by their word count, so a 250-word chunk outvotes a
    40-word one, and each contributes ``confidence`` as a multiplier so a
    coin-flip window cannot outvote a decisive one. A true 40/60 mix is
    reported as mixed with both codes instead of hiding the minority language.
    """
    weight: dict[str, float] = {}
    words: Counter = Counter()
    for r in results or []:
        code = (r or {}).get("code") or ""
        if not code:
            continue
        w = max(1, int(r.get("words") or 1)) * max(0.15, float(r.get("confidence") or 0.3))
        weight[code] = weight.get(code, 0.0) + w
        words[code] += max(1, int(r.get("words") or 1))
    if not weight:
        return {"code": "", "name": "Unknown", "confidence": 0.0, "mixed": False,
                "secondary": None, "languages": []}
    total = sum(weight.values())
    ranked = sorted(weight.items(), key=lambda kv: kv[1], reverse=True)
    top, top_w = ranked[0]
    second_w = ranked[1][1] if len(ranked) > 1 else 0.0
    share = top_w / total
    languages = [{"code": c, "name": LANGUAGES.get(c, (c, c, False))[0],
                  "share": round(w / total, 3), "words": words[c]}
                 for c, w in ranked if w / total >= 0.05]
    mixed = bool(second_w / total >= 0.25)
    name, _native, _rtl = LANGUAGES.get(top, (top.upper(), top, False))
    return {
        "code": top, "name": name,
        "confidence": round(min(0.99, share * (1.0 if not mixed else 0.7)), 3),
        "mixed": mixed,
        "secondary": ranked[1][0] if mixed else None,
        "languages": languages,
    }


def label(code: str | None) -> str:
    if not code:
        return "Unknown"
    return LANGUAGES.get(code, (code.upper(), code, False))[0]


def info() -> dict:
    return {"engine": "builtin-function-words", "languages": len(LANGUAGES),
            "codes": sorted(LANGUAGES), "min_words": _MIN_WORDS}
