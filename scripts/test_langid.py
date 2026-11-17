"""Language-identification regression test — pure text, no network.

Each fixture is realistic spoken-style text (what the ASR actually emits), not
dictionary examples: short sentences, filler words, mixed casing.

Run from the repo root:  .venv/bin/python scripts/test_langid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscribe import langid  # noqa: E402

FIXTURES: list[tuple[str | tuple[str, ...], str]] = [
    ("en", "Good morning everyone, thanks for joining the call on such short notice. "
           "I had a look at the numbers and I think we should ship the migration next week, "
           "but we need to check whether the load testing is done."),
    ("de", "Guten Morgen zusammen, danke dass ihr so kurzfristig dabei seid. Ich habe mir die "
           "Zahlen angeschaut und ich denke, wir sollten die Migration nächste Woche ausrollen, "
           "aber wir müssen noch prüfen, ob die Lasttests fertig sind und ob der Antrag durch ist."),
    ("fr", "Bonjour à tous, merci d'être là malgré le délai très court. J'ai regardé les chiffres "
           "et je pense que nous devrions livrer la migration la semaine prochaine, mais il faut "
           "vérifier si les tests de charge sont terminés et si le contrat est signé."),
    ("es", "Buenos días a todos, gracias por venir con tan poco tiempo. He revisado los números y "
           "creo que deberíamos lanzar la migración la próxima semana, pero tenemos que comprobar "
           "si las pruebas de carga están terminadas y si el contrato está firmado."),
    ("it", "Buongiorno a tutti, grazie per essere qui con così poco preavviso. Ho guardato i numeri "
           "e penso che dovremmo rilasciare la migrazione la prossima settimana, ma dobbiamo "
           "verificare se i test di carico sono finiti e se il contratto è stato firmato."),
    ("pt", "Bom dia a todos, obrigado por virem com tão pouco tempo. Eu olhei os números e acho que "
           "devemos lançar a migração na próxima semana, mas precisamos verificar se os testes de "
           "carga estão concluídos e se o contrato foi assinado."),
    ("nl", "Goedemorgen allemaal, bedankt dat jullie zo snel konden komen. Ik heb naar de cijfers "
           "gekeken en ik denk dat we de migratie volgende week moeten uitrollen, maar we moeten "
           "nog controleren of de belastingstests klaar zijn en of het contract is ondertekend."),
    ("sv", "God morgon allihopa, tack för att ni kunde komma med så kort varsel. Jag har tittat på "
           "siffrorna och jag tycker att vi borde rulla ut migreringen nästa vecka, men vi måste "
           "kontrollera om lasttesterna är klara och om avtalet är påskrivet."),
    ("da", "Godmorgen alle sammen, tak fordi I kunne komme med så kort varsel. Jeg har kigget på "
           "tallene, og jeg synes, vi skal udrulle migreringen i næste uge, men vi skal kontrollere, "
           "om belastningstestene er færdige, og om kontrakten er underskrevet."),
    ("nb", "God morgen alle sammen, takk for at dere kunne komme med så kort varsel. Jeg har sett på "
           "tallene, og jeg synes vi bør rulle ut migreringen neste uke, men vi må kontrollere om "
           "belastningstestene er ferdige og om kontrakten er signert."),
    ("fi", "Hyvää huomenta kaikille, kiitos että pääsitte näin lyhyellä varoitusajalla. Katsoin "
           "lukuja ja mielestäni meidän pitäisi julkaista migraatio ensi viikolla, mutta meidän "
           "täytyy vielä tarkistaa, ovatko kuormitustestit valmiita ja onko sopimus allekirjoitettu."),
    ("pl", "Dzień dobry wszystkim, dziękuję że mogliście przyjść tak szybko. Sprawdziłem liczby i "
           "uważam, że powinniśmy wdrożyć migrację w przyszłym tygodniu, ale musimy jeszcze "
           "sprawdzić, czy testy obciążeniowe są gotowe i czy umowa jest podpisana."),
    ("cs", "Dobré ráno všem, děkuji že jste mohli přijít tak rychle. Podíval jsem se na čísla a "
           "myslím, že bychom měli migraci nasadit příští týden, ale musíme ještě zkontrolovat, "
           "jestli jsou zátěžové testy hotové a jestli je smlouva podepsaná."),
    ("hu", "Jó reggelt mindenkinek, köszönöm hogy ilyen gyorsan eljöttek. Megnéztem a számokat és "
           "szerintem jövő héten kellene telepíteni a migrációt, de még ellenőrizni kell, hogy a "
           "terhelési tesztek készen vannak-e és alá van-e írva a szerződés."),
    ("ro", "Bună dimineața tuturor, mulțumesc că ați putut veni atât de repede. M-am uitat la cifre "
           "și cred că ar trebui să livrăm migrarea săptămâna viitoare, dar trebuie să verificăm "
           "dacă testele de încărcare sunt gata și dacă contractul este semnat."),
    ("tr", "Herkese günaydın, bu kadar kısa sürede gelebildiğiniz için teşekkürler. Rakamlara "
           "baktım ve bence göçü gelecek hafta yayınlamalıyız ama yük testlerinin bitip "
           "bitmediğini ve sözleşmenin imzalanıp imzalanmadığını kontrol etmemiz gerekiyor."),
    ("ru", "Доброе утро всем, спасибо что смогли прийти так быстро. Я посмотрел цифры и думаю, что "
           "мы должны выпустить миграцию на следующей неделе, но нужно проверить, готовы ли "
           "нагрузочные тесты и подписан ли договор."),
    ("uk", "Доброго ранку всім, дякую що змогли прийти так швидко. Я подивився цифри і думаю, що ми "
           "повинні випустити міграцію наступного тижня, але треба перевірити, чи готові "
           "навантажувальні тести і чи підписаний договір."),
    ("bg", "Добро утро на всички, благодаря че успяхте да дойдете толкова бързо. Прегледах числата и "
           "мисля, че трябва да пуснем миграцията следващата седмица, но трябва да проверим дали "
           "тестовете за натоварване са готови и дали договорът е подписан."),
    ("el", "Καλημέρα σε όλους, ευχαριστώ που ήρθατε τόσο γρήγορα. Κοίταξα τους αριθμούς και νομίζω "
           "ότι πρέπει να κυκλοφορήσουμε τη μετάβαση την επόμενη εβδομάδα, αλλά πρέπει να "
           "ελέγξουμε αν οι δοκιμές φόρτωσης έχουν τελειώσει."),
    ("ar", "صباح الخير للجميع، شكرا لأنكم تمكنتم من الحضور بهذه السرعة. نظرت في الأرقام وأعتقد أننا "
           "يجب أن ننشر الترحيل الأسبوع المقبل، لكن علينا التحقق من أن اختبارات التحميل جاهزة."),
    ("he", "בוקר טוב לכולם, תודה שהצלחתם להגיע כל כך מהר. הסתכלתי על המספרים ואני חושב שאנחנו "
           "צריכים לשחרר את ההעברה בשבוע הבא, אבל אנחנו חייבים לבדוק שהבדיקות מוכנות."),
    ("zh", "大家早上好，谢谢你们这么快就来了。我看了这些数字，我认为我们应该在下周发布迁移，"
           "但是我们需要确认负载测试是否已经完成，以及合同是否已经签署。"),
    ("ja", "皆さんおはようございます、急なお願いにもかかわらず集まっていただきありがとうございます。"
           "数字を確認しましたが、来週マイグレーションをリリースすべきだと思います。ただし負荷テストが"
           "終わっているか確認する必要があります。"),
    ("ko", "모두 안녕하세요, 이렇게 급하게 와 주셔서 감사합니다. 숫자를 살펴보았는데 다음 주에 "
           "마이그레이션을 배포해야 한다고 생각합니다. 다만 부하 테스트가 끝났는지 확인해야 합니다."),
    ("hi", "सभी को सुप्रभात, इतनी जल्दी आने के लिए धन्यवाद। मैंने संख्याएं देखीं और मुझे लगता है कि "
           "हमें अगले हफ्ते माइग्रेशन जारी करना चाहिए, लेकिन हमें जांचना होगा कि लोड टेस्ट पूरे हुए हैं।"),
    ("th", "สวัสดีตอนเช้าทุกคน ขอบคุณที่มาทันเวลาขนาดนี้ ผมดูตัวเลขแล้วคิดว่าเราควรปล่อยการย้ายระบบ"
           "สัปดาห์หน้า แต่เราต้องตรวจสอบว่าการทดสอบโหลดเสร็จแล้วหรือยัง"),
    # Code-switched sentence: English holds most of the words, so either answer
    # is defensible — what must not happen is losing the second language.
    (("en", "de"), "Also der Bescheid ist noch nicht da, but I think the deadline is next Friday, "
                   "so we should escalate this with the Abteilung before the Landeshauptmann asks again."),
]

SHORT_CASES = [
    ("de", "Ja, das ist gut."),
    ("en", "No, I think so."),
    ("fr", "Oui, c'est très bien."),
]

MIXED_CASE = (
    "Wir haben die Migration am Montag gestartet und die Zahlen sehen gut aus. "
    "The load tests are still running and I think we need two more weeks before we can ship. "
    "Der Antrag liegt beim Magistrat und wird nächste Woche bearbeitet. "
    "Can you confirm whether the contract has been signed by the vendor? "
    "Die Abteilung hat noch keine Rückmeldung gegeben, wir müssen nachfassen."
)


def main() -> int:
    failures = 0
    print(f"{'expected':<10}{'got':<10}{'conf':<7}{'words':<7}text")
    print("-" * 96)
    for want, text in FIXTURES:
        r = langid.detect(text)
        accepted = (want,) if isinstance(want, str) else want
        ok = r["code"] in accepted
        failures += 0 if ok else 1
        mark = " " if ok else "✗"
        print(f"{'/'.join(accepted):<10}{r['code'] or '—':<10}{r['confidence']:<7.2f}"
              f"{r['words']:<7}{mark} {text[:52]}…")
        if not ok:
            print(f"           alternatives: {r['alternatives']}")
        if len(accepted) > 1 and r["code"] and not r["mixed"]:
            failures += 1
            print(f"           ✗ code-switched text not flagged mixed (secondary={r['secondary']})")

    print("\nshort utterances (low confidence is acceptable, wrong code is not):")
    for want, text in SHORT_CASES:
        r = langid.detect(text)
        ok = r["code"] == want
        failures += 0 if ok else 1
        print(f"  {'ok ' if ok else '✗  '}{want} -> {r['code'] or '—'} "
              f"({r['confidence']:.2f})  {text!r}")

    mixed = langid.detect(MIXED_CASE)
    doc = langid.merge([langid.detect(s) for s in MIXED_CASE.split(". ")])
    print(f"\nbilingual paragraph: detect -> {mixed['code']} ({mixed['confidence']:.2f}), "
          f"rollup -> {doc['code']} mixed={doc['mixed']} secondary={doc['secondary']}")
    if not doc["code"]:
        failures += 1

    # A window of pure noise must not invent a language.
    noise = langid.detect("Mm. Hmm. Uh. Yeah. Okay.")
    print(f"filler only -> code={noise['code']!r} conf={noise['confidence']:.2f} "
          f"(expected low or empty)")
    if noise["code"] and noise["confidence"] > 0.55:
        failures += 1
        print("  ✗ overconfident on filler")

    print("\nPASS" if not failures else f"\nFAIL ({failures} case(s))")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
