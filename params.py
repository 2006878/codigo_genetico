CAIXA_INICIAL =  10000
CAPITAL_DE_GIRO = 5000
LIMITE_CAIXA = 40000  # O valor máximo que você pode ficar no caixa
MARKUP = 100  # em %

PRAZO_MEDIO_ESTOCAGEM = 1  # em meses
PRAZO_MEDIO_RECEBIMENTO = 2
PRAZO_MEDIO_PAGAMENTO = 3

CUSTO_OPORTUNIDADE = 0.01  # em porcentagem

CUSTO_TRANSACAO = {'fixo': 30,
                   'percentual': 0.002}

CICLO_MEDIO_CAIXA = PRAZO_MEDIO_ESTOCAGEM + PRAZO_MEDIO_RECEBIMENTO - PRAZO_MEDIO_PAGAMENTO

if CICLO_MEDIO_CAIXA < 0:
    CICLO_MEDIO_CAIXA = 0

PROVISIONAMENTO = {'juros': 0.005,
                   'n_dias': 10}

N_DIAS = 30

RECEBIMENTOS = [ 3000, 1000, 2000, 4000, 1500, 4000, 2000, 5000, 3500, 3700, 1900, 1800, 5000, 2000, 8000, 1900, 2900, 3200, 1900, 4500, 2700, 2000, 3900, 2300, 3200, 3500, 1800, 4800, 2000, 3000, ]

# 5 dia é dia de pagamento
PAGAMENTOS =   [ 2000, 3000, 2000, 3000, 21000, 1400, 1800, 4000, 2500, 2200, 2200, 1000,  500, 1000, 7000, 1300, 1800, 2600, 17800, 2800, 2900, 1100, 2800, 2700, 2800, 2000, 1000, 1900, 2000, 3000, ]
