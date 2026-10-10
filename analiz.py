"""E-posta kampanyası deneyi: kampanya satış getirdi mi, kime işe yaradı, hedefleme işe yarar mı?

Veri: Kevin Hillstrom, MineThatData E-Mail Analytics Challenge (2008). 64.000 müşteri rastgele üç gruba ayrılmış:
erkek ürünleri e-postası, kadın ürünleri e-postası, e-posta yok. Sonuçlar e-postadan sonraki iki haftayı kapsar.

Girdi : data/hillstrom.csv
Çıktı : outputs/summary.json, outputs/*.png

Kurulum: pip install pandas numpy scipy scikit-learn matplotlib
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

SEED = 42
B = 2000  # bootstrap tekrarı
OUT = Path("outputs")
GRUP = {"No E-Mail": "yok", "Mens E-Mail": "erkek", "Womens E-Mail": "kadin"}
KOLLAR = ["yok", "erkek", "kadin"]
SONUC = ["visit", "conversion", "spend"]
ACCENT, GRAY, WARM = "#6d28d9", "#8a8799", "#d95926"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e1ec", "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10})


def y(x, n=4):
    return round(float(x), n)


def fark(a: pd.Series, b: pd.Series) -> dict:
    """İki bağımsız grubun ortalama farkı, %95 aralık (normal yaklaşım) ve p değeri."""
    f = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return {"fark": y(f), "alt": y(f - 1.96 * se), "ust": y(f + 1.96 * se), "se": y(se, 5),
            "p": float(2 * (1 - stats.norm.cdf(abs(f / se)))), "goreli": y(f / b.mean() * 100, 1) if b.mean() else None,
            "n_a": int(len(a)), "n_b": int(len(b))}


def oku() -> pd.DataFrame:
    d = pd.read_csv("data/hillstrom.csv")
    d["grup"] = d["segment"].map(GRUP)
    assert d["grup"].notna().all() and len(d) == 64000
    # geçmiş alışveriş kategorisi
    d["gecmis"] = np.select([(d.mens == 1) & (d.womens == 0), (d.mens == 0) & (d.womens == 1)], ["erkek", "kadin"], "ikisi")
    return d


def deney_kontrolu(d: pd.DataFrame) -> dict:
    n = d["grup"].value_counts()
    srm = stats.chisquare(n.values).pvalue
    # standartlaştırılmış ortalama farkı: her grup, e-posta almayanlarla
    X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]], columns=["zip_code", "channel"], dtype=float)
    smd = {}
    for g in ("erkek", "kadin"):
        a, b = X[d.grup == g], X[d.grup == "yok"]
        s = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        smd[g] = {k: y(v, 3) for k, v in s.items()}
    en_buyuk = max(abs(v) for g in smd.values() for v in g.values())
    return {"n": {k: int(v) for k, v in n.items()}, "srm_p": y(srm, 3), "smd": smd, "en_buyuk_smd": y(en_buyuk, 3)}


def ortalama_etki(d: pd.DataFrame) -> dict:
    c = d[d.grup == "yok"]
    R = {"oran": {g: {s: y(d.loc[d.grup == g, s].mean()) for s in SONUC} for g in KOLLAR}}
    R["etki"] = {g: {s: fark(d.loc[d.grup == g, s], c[s]) for s in SONUC} for g in ("erkek", "kadin")}
    R["erkek_kadin"] = {s: fark(d.loc[d.grup == "erkek", s], d.loc[d.grup == "kadin", s]) for s in SONUC}
    alan = d[d.conversion == 1]
    R["sepet"] = {g: {"ort": y(alan.loc[alan.grup == g, "spend"].mean(), 2), "n": int((alan.grup == g).sum())} for g in KOLLAR}
    R["sepet_fark"] = fark(alan.loc[alan.grup == "erkek", "spend"], alan.loc[alan.grup == "yok", "spend"])
    R["alici"] = int(d.conversion.sum())
    # Atıf: e-posta alanların harcamasının ne kadarı e-posta olmadan da olacaktı?
    R["artimsal_pay"] = {g: y(R["etki"][g]["spend"]["fark"] / R["oran"][g]["spend"] * 100, 1) for g in ("erkek", "kadin")}
    # Etkinin kaynağı: ziyaret mi, ziyaretten satışa geçiş mi?
    R["ziyaret_satis"] = {g: y(d.loc[(d.grup == g) & (d.visit == 1), "conversion"].mean() * 100, 2) for g in KOLLAR}
    return R


def alt_gruplar(d: pd.DataFrame) -> dict:
    """Önceden seçilmiş birkaç değişkene göre etki. Çoklu karşılaştırma var; sonuçlar ipucu olarak okunmalı."""
    d = d.assign(yenilik=pd.cut(d.recency, [0, 3, 6, 9, 12], labels=["1–3 ay", "4–6 ay", "7–9 ay", "10–12 ay"]).astype(str),
                 harcama=pd.cut(d.history, [0, 100, 200, 500, 1e9], labels=["<$100", "$100–200", "$200–500", "$500+"], right=False).astype(str))
    R = {}
    for v in ("gecmis", "newbie", "channel", "zip_code", "yenilik", "harcama"):
        R[v] = {}
        for k, s in d.groupby(v):
            c = s[s.grup == "yok"]
            R[v][str(k)] = {"n": int(len(s)), **{g: {o: fark(s.loc[s.grup == g, o], c[o]) for o in ("visit", "spend")} for g in ("erkek", "kadin")}}
    # Etkileşim testi: kadın e-postasının etkisi, yalnızca erkek ürünü alanlarla yalnızca kadın ürünü alanlar arasında farklı mı?
    test = {}
    for g in ("erkek", "kadin"):
        for o in ("visit", "spend"):
            a, b = R["gecmis"]["kadin"][g][o], R["gecmis"]["erkek"][g][o]
            f = a["fark"] - b["fark"]
            se = np.sqrt(a["se"] ** 2 + b["se"] ** 2)
            test[f"{g}_{o}"] = {"fark": y(f), "alt": y(f - 1.96 * se), "ust": y(f + 1.96 * se), "p": float(2 * (1 - stats.norm.cdf(abs(f / se))))}
    R["etkilesim_kadin_erkek_gecmis"] = test
    return R


def hedefleme(d: pd.DataFrame) -> dict:
    """Üç kollu T-learner: her grup için ayrı model, 5 katlı çapraz uydurma (her müşteri, görmediği modellerle puanlanır)."""
    X = pd.get_dummies(d[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]], columns=["zip_code", "channel"], dtype=float)
    P = {(a, s): np.zeros(len(d)) for a in KOLLAR for s in ("visit", "spend")}
    katlar = StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, d.grup + d.conversion.astype(str))
    for tr, te in katlar:
        for a in KOLLAR:
            m = tr[d.grup.values[tr] == a]
            c = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=100, random_state=SEED)
            P[(a, "visit")][te] = c.fit(X.iloc[m], d.visit.iloc[m]).predict_proba(X.iloc[te])[:, 1]
            r = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=SEED)
            P[(a, "spend")][te] = r.fit(X.iloc[m], d.spend.iloc[m]).predict(X.iloc[te])

    kollar = np.array(KOLLAR)
    politika = {"kimseye": np.full(len(d), "yok"), "herkese_erkek": np.full(len(d), "erkek"), "herkese_kadin": np.full(len(d), "kadin"),
                "kural": np.where(d.gecmis == "kadin", "kadin", "erkek"),
                "model_ziyaret": kollar[np.column_stack([P[(a, "visit")] for a in KOLLAR]).argmax(1)],
                "model_harcama": kollar[np.column_stack([P[(a, "spend")] for a in KOLLAR]).argmax(1)]}
    # Rastgele atama her kol için 1/3 olasılıklı olduğundan, bir politikanın değeri: önerdiği kolu tesadüfen almış müşterilerin ortalaması.
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    grup = d.grup.values
    R = {}
    for ad, p in politika.items():
        es = grup == p
        R[ad] = {"pay": {a: y((p == a).mean(), 3) for a in KOLLAR}, "eslesen": int(es.sum())}
        for s in ("visit", "spend"):
            v = d[s].values
            boot = np.array([v[i][es[i]].mean() for i in idx[:500]])
            R[ad][s] = {"deger": y(v[es].mean()), "alt": y(np.percentile(boot, 2.5)), "ust": y(np.percentile(boot, 97.5))}
    # Model, "herkese erkek"ten ne kadar farklı? Eşleştirilmiş bootstrap (aynı örneklemde iki politika)
    farklar = {}
    for ad in ("kural", "model_ziyaret", "model_harcama"):
        a_es, b_es = grup == politika[ad], grup == "erkek"
        for s in ("visit", "spend"):
            v = d[s].values
            bs = np.array([v[i][a_es[i]].mean() - v[i][b_es[i]].mean() for i in idx[:500]])
            farklar[f"{ad}_{s}"] = {"fark": y(v[a_es].mean() - v[b_es].mean()), "alt": y(np.percentile(bs, 2.5)), "ust": y(np.percentile(bs, 97.5))}
    R["herkese_erkekten_fark"] = farklar

    # Artış eğrisi: erkek e-postası, tahmini etkisi en yüksek müşteriden başlayarak gönderilseydi
    egri = {}
    for s in ("visit", "spend"):
        u = P[("erkek", s)] - P[("yok", s)]
        sira = np.argsort(-u, kind="stable")
        toplam = None
        noktalar = []
        for q in np.linspace(0.1, 1.0, 10):
            ust = sira[: int(round(q * len(d)))]
            g, v = grup[ust], d[s].values[ust]
            artis = (v[g == "erkek"].mean() - v[g == "yok"].mean()) * len(ust)
            noktalar.append(artis)
        toplam = noktalar[-1]
        egri[s] = [y(v / toplam * 100, 1) for v in noktalar]
    R["artis_egrisi"] = egri
    return R


def ekonomi(R: dict) -> dict:
    """Maliyet ve kâr marjı veride yok; başabaş maliyetini farklı marj varsayımlarıyla hesapla."""
    out = {}
    for g in ("erkek", "kadin"):
        e = R["etki"][g]["spend"]
        out[g] = {str(m): {"basabas": y(e["fark"] * m / 100, 3), "alt": y(e["alt"] * m / 100, 3), "ust": y(e["ust"] * m / 100, 3)} for m in (20, 30, 40, 50)}
    return out


def orneklem(d: pd.DataFrame) -> dict:
    """İki e-posta arasındaki harcama farkını %80 güçle görmek için grup başına kaç müşteri gerekir?"""
    sd = d.spend.std()
    out = {"sd": y(sd, 2)}
    for f in (0.35, 0.77, 0.25):
        out[str(f)] = int(np.ceil(2 * (1.96 + 0.8416) ** 2 * sd ** 2 / f ** 2))
    return out


def grafikler(S: dict) -> None:
    E = S["etki"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, s, ad, carp in zip(axes, SONUC, ("Siteyi ziyaret (%)", "Alışveriş (%)", "Müşteri başına harcama ($)"), (100, 100, 1)):
        vals = [S["oran"][g][s] * carp for g in KOLLAR]
        ax.bar(["E-posta yok", "Erkek", "Kadın"], vals, color=[GRAY, ACCENT, ACCENT], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if s == "spend" else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(ad, fontsize=10); ax.grid(axis="x", visible=False)
    plt.tight_layout(); fig.savefig(OUT / "1_gruplar.png"); plt.close(fig)

    A = S["alt_gruplar"]["gecmis"]
    fig, ax = plt.subplots(figsize=(7, 3))
    etiket = {"erkek": "Yalnızca erkek ürünü almış", "kadin": "Yalnızca kadın ürünü almış", "ikisi": "İkisini de almış"}
    ks = ["erkek", "kadin", "ikisi"]
    for j, (g, renk, kayma) in enumerate((("erkek", ACCENT, -0.15), ("kadin", WARM, 0.15))):
        f = [A[k][g]["visit"]["fark"] * 100 for k in ks]
        e = [(A[k][g]["visit"]["ust"] - A[k][g]["visit"]["fark"]) * 100 for k in ks]
        ax.errorbar(np.arange(3) + kayma, f, yerr=e, fmt="o", color=renk, capsize=3, label=f"{'Erkek' if g == 'erkek' else 'Kadın'} e-postası")
    ax.axhline(0, color="#c9c5d6")
    ax.set(xticks=range(3), xticklabels=[etiket[k] for k in ks], ylabel="Ziyaret oranına etki (puan)", title="E-posta kime işe yaradı?")
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "2_gecmis.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.linspace(10, 100, 10)
    ax.plot([0, 100], [0, 100], color="#c9c5d6", linestyle="--", label="Rastgele seçim")
    ax.plot(np.r_[0, x], np.r_[0, S["hedefleme"]["artis_egrisi"]["visit"]], marker="o", color=ACCENT, label="Model: ek ziyaret")
    ax.plot(np.r_[0, x], np.r_[0, S["hedefleme"]["artis_egrisi"]["spend"]], marker="o", color=WARM, label="Model: ek harcama")
    ax.set(xlabel="E-posta gönderilen müşteri payı (%)", ylabel="Yakalanan toplam etki (%)", title="Model hedeflemesi rastgeleden iyi mi?")
    ax.legend(frameon=False)
    plt.tight_layout(); fig.savefig(OUT / "3_artis_egrisi.png"); plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    d = oku()
    S = {"veri": {"musteri": int(len(d)), "alici": int(d.conversion.sum()), "ziyaret": int(d.visit.sum()), "toplam_harcama": y(d.spend.sum(), 2)}}
    S["deney"] = deney_kontrolu(d)
    S.update(ortalama_etki(d))
    S["alt_gruplar"] = alt_gruplar(d)
    S["hedefleme"] = hedefleme(d)
    S["ekonomi"] = ekonomi(S)
    S["orneklem"] = orneklem(d)
    (OUT / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    grafikler(S)
    print(json.dumps({k: S[k] for k in ("veri", "deney", "oran", "artimsal_pay", "sepet", "ziyaret_satis", "ekonomi", "orneklem")}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in S["hedefleme"].items()}, ensure_ascii=False))
    print(json.dumps(S["alt_gruplar"]["etkilesim_kadin_erkek_gecmis"], ensure_ascii=False))


if __name__ == "__main__":
    main()
