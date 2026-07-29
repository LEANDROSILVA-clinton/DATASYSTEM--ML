from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


# abaixo,voce tem o uso de SVM usado para regressão


# dados brutos: cancelamentos e vendas totais por semana
jan_cancel = [259.70, 345.50, 518.00, 745.00]
jan_total_semana = [4599.90, 1798.00, 1299.00, 799.60]

fev_cancel = [359.40, 200.00, 798.00, 359.00]
fev_total_semana = [2199.00, 479.70, 1399.00, 698.00]

PADRAO = 0.60  # teto de 60% que pode haver de cancelamentos ,passou disso ,você tem uma queda drastica


def taxa_semanal(cancel, total):
    return [c/t for c, t in zip(cancel, total)]


def desvio_padrao(taxas, padrao=PADRAO):
    # funciona dessa forma, acima do padrão = ruim , abaixo do teto = bom resultado
    return [t - padrao for t in taxas]


jan_taxas = taxa_semanal(jan_cancel, jan_total_semana)
fev_taxas = taxa_semanal(fev_cancel, fev_total_semana)

jan_desvio = desvio_padrao(jan_taxas)
fev_desvio = desvio_padrao(fev_taxas)

print("JANEIRO")
for i, (t, d) in enumerate(zip(jan_taxas, jan_desvio), 1):
    status = "Acima do Padrão" if d > 0 else "dentro do padrão"
    print(f"Semana{i}: taxa={t*100:.2f}% desvio = {d*100:+.2f}pp ({status})")

print("\nFEVEREIRO")
for i, (t, d) in enumerate(zip(fev_taxas, fev_desvio), 1):
    status = "Acima do Padrão" if d > 0 else "dentro do padrão"
    print(f"Semana{i}: taxa={t*100:.2f}% desvio = {d*100:+.2f}pp ({status})")

    print(
        f"\nMédia desvio Jan:{np.mean(jan_desvio)*100:+.2f}pp|Desvio Padrão Jan:{np.std(jan_taxas)*100:.2f}pp")
    print(
        f"Média Desvio Fev:{np.mean(fev_desvio)*100:+.2f}pp|Desvio Padrão Fev:{np.std(fev_taxas)*100:.2f}pp")

    # Regressão SVR logo abaixo(mais informações checar doc oficial scikit-learn 1.4.2)
    # x = índice da semana (contínuo, jan = semanas 1-4, fev= semanas 5-8)
    # y = taxa de cancelamento observada

    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    y = np.array(jan_taxas + fev_taxas)

    # abaixo ,foi recomendado pela propría documentação : escalar os dados antes do uso da regressão SVR
    modelo = make_pipeline(StandardScaler(), SVR(
        kernel='rbf', C=1.0, epsilon=0.05))
    modelo.fit(x, y)

    pred = modelo.predict(x)
    print("\n===SVR: taxa prevista vs observada===")
    for i, (real, p) in enumerate(zip(y, pred), 1):
        print(f"Semana {i}: real = {real*100:.2f}% previsto(SVR)={p*100:.2f}%")

        # previsão para a proxima semana, um teste simples
        proxima = modelo.predict([[9]])

        print(
            f"\nPrevisão SVR para semana 9(próxima semana): {proxima[0]*100:.2f}%")
        print(
            f"Desvio previsto em relação ao padrão de 60%: {(proxima[0]-PADRAO)*100:+.2f}pp")

# scope de anomalia = detecção de outlier/LocalOutlierFactor, não supervisionada

jan_cancel = [259.70, 345.50, 518.00, 745.00]
jan_total = [4599.90, 1798.00, 1299.00, 799.60]
fev_cancel = [359.40, 899.50, 798.00, 359.00]
fev_total = [2199.00, 479.70, 1399.00, 698.00]

cancel = jan_cancel + fev_cancel
total = jan_total + fev_total
taxa = [c/t for c, t in zip(cancel, total)]

labels = ["Jan-S1", "Jan-S2", "Jan-S3", "Jan-S4",
          "Fev-S1", "Fev-S2", "Fev-S3", "Fev-S4"]

# Features: [cancelamento, venda total e taxa de cancelamento]

x = np.array(list(zip(cancel, total, taxa)))


# escalamento das features (duvidas ,cheque a doc oficial do scikit-learn em pratica padrao)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# n_neighbors=20 é o padrao ou default seguindo a documentação,mas como tem apenas 8 amostras acontece uma redução simples
lof = LocalOutlierFactor(n_neighbors=3, contamination='auto')
# definmos como 1=normal, -1 = abaixo do padrao
pred = lof.fit_predict(x_scaled)
scores = lof.negative_outlier_factor_  # funçaõ de quanto mais negativo ,pior
print("Resulatdo LOF(detecção de anomalias)")
for lbl, c, t, tx, p, s in zip(labels, cancel, total, taxa, pred, scores):
    status = "Anomalia" if p == -1 else "normal"
    print(f"{lbl}:cancel=R${c:.2f} total =R${t:.2f} taxa = {tx*100:.2f}% score = {s:.3f} ->{status}")
