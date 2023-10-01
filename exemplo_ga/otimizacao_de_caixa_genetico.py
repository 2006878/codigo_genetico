"""
Previsão de fluxo de caixa durante 30 dias.

##########################
#  Opções de empréstimo  #
##########################
Empréstimo 1: 0.2 juros / dias: 3
Empréstimo 2: 0.1 juros / dias: 5

############################
#  Opções de investimento  #
############################
Investimento 1: 0.1 juros / dias: 3
Investimento 2: 0.2 juros / dias: 5

############
#  Regras  #
############
* Se o caixa estiver positivo, um investimento é realizado
* Se o caixa estiver negativo, um empréstimo é realizado

30 dias * 2 dimensões (opcao_emprestimo, opcao_investimento)
cromossomo de comprimento 60
"""
import random
import copy
import numpy as np
from parametros import retorno_diario, emprestimo, investimento
from parametros import N_GERACOES, N_CROMOSSOMOS, TOTAL_DIAS, PROB_MUTACAO


N_VARS_DIA = 2

NENHUMA = 0
EMPRESTIMO = 1
INVESTIMENTO = 2


# Conjunto de ações realizadas
informacoes_diarias = []


def adicionar_info_dia(acao, tipo, entrada, saida):
    informacoes_diarias.append(
        {"ação": acao, "tipo": tipo, "entrada": entrada, "saida": saida}
    )


def calcular_fitness(cromossomo, registrar_acoes=False):
    balanco = 0
    copia_retorno_diario = copy.deepcopy(retorno_diario)

    for dia in range(TOTAL_DIAS):
        entrada_dia = float(copia_retorno_diario[dia]['entrada'])
        saida_dia = float(copia_retorno_diario[dia]['saida'])

        balanco = (entrada_dia - saida_dia)

        opcao_emprestimo = cromossomo[dia * N_VARS_DIA]
        opcao_investimento = cromossomo[dia * N_VARS_DIA + 1]

        n_dias_emprestimo = emprestimo[opcao_emprestimo]['dias']
        n_dias_investimento = investimento[opcao_investimento]['dias']

        if balanco < 0 and dia + n_dias_emprestimo < TOTAL_DIAS:
            opcao_emprestimo = cromossomo[dia * N_VARS_DIA]

            juros = emprestimo[opcao_emprestimo]['juros']

            dia_pagamento = dia + n_dias_emprestimo

            copia_retorno_diario[dia_pagamento]['saida'] += (juros * n_dias_emprestimo + 1) * balanco
            if registrar_acoes:
                adicionar_info_dia(EMPRESTIMO, opcao_emprestimo,
                                   entrada_dia, saida_dia)
            balanco = 0

        elif balanco > 0 and dia + n_dias_investimento < TOTAL_DIAS:
            opcao_investimento = cromossomo[dia * N_VARS_DIA + 1]

            juros = investimento[opcao_investimento]['juros']

            dia_recebimento = dia + n_dias_investimento

            copia_retorno_diario[dia_recebimento]['entrada'] += (juros * n_dias_investimento + 1) * balanco
            if registrar_acoes:
                adicionar_info_dia(INVESTIMENTO, opcao_investimento,
                                   entrada_dia, saida_dia)

            balanco = 0

        elif registrar_acoes:
            adicionar_info_dia(NENHUMA, 0, entrada_dia, saida_dia)

    return balanco


def selecionar_pais(fitness_cromossomos):
    if max(fitness_cromossomos) > min(fitness_cromossomos):
        fitness_cromossomos = (fitness_cromossomos - min(fitness_cromossomos)) / (
                                max(fitness_cromossomos) - min(fitness_cromossomos))

        roleta = np.cumsum(fitness_cromossomos)
        roleta /= max(roleta)
    else:
        roleta = np.ones(fitness_cromossomos.shape[0]) / fitness_cromossomos.shape[0]

    sorteio = random.random()
    pai1 = np.argmin(roleta < sorteio)

    sorteio = random.random()
    pai2 = np.argmin(roleta < sorteio)
    return pai1, pai2


def mutacao(cromossomos):
    sorteios = np.random.random(cromossomos.shape[0])
    indices = np.where(sorteios < PROB_MUTACAO)[0]
    posicoes = np.random.randint(cromossomos.shape[1], size=indices.shape[0])
    cromossomos[indices, posicoes] = 1 - cromossomos[indices, posicoes]


def gerar_solucoes_iniciais():
    cromossomos = np.random.randint(
        0, 2, (N_CROMOSSOMOS, TOTAL_DIAS * N_VARS_DIA)
    )
    return cromossomos


def crossover(cromossomos, fitness_cromossomos):
    filhos = []
    for _ in range(int(N_CROMOSSOMOS / 2)):

        pai1, pai2 = selecionar_pais(fitness_cromossomos)
        posicao = random.randint(1, cromossomos.shape[1]-2)

        filho1 = np.copy(cromossomos[pai1])
        filho2 = np.copy(cromossomos[pai2])

        aux = np.copy(filho1[0:posicao])
        filho1[0:posicao] =  filho2[0:posicao]
        filho2[0:posicao] = aux

        filhos.append(filho1)
        filhos.append(filho2)
    return np.array(filhos)


def mostrar_solucao():
    for dia, info_dia in enumerate(informacoes_diarias):
        print("--------------- Dia %d ---------------" % (dia+1))
        if info_dia['ação'] == EMPRESTIMO:
            n_dias = emprestimo[info_dia['tipo']]['dias']
            juros = emprestimo[info_dia['tipo']]['juros']
            print("Ação realizada: Empréstimo.")
            print("%.1f de juros por %d dias." % (juros, n_dias))

        elif info_dia['ação'] == INVESTIMENTO:
            n_dias = investimento[info_dia['tipo']]['dias']
            juros = investimento[info_dia['tipo']]['juros']
            print("Ação realizada: Investimento.")
            print("%.1f de juros por %d dias." % (juros, n_dias))

        else:
            print("Nenhuma ação realizada.")
        print("Entrada do caixa: R$ %.2f" % info_dia["entrada"])
        print("Saída do caixa: R$ %.2f" % info_dia["saida"])
        balanco = info_dia["entrada"] - info_dia["saida"]
        print("Balanço no final do dia: R$ %.2f" % balanco)


def main():
    cromossomos = gerar_solucoes_iniciais()
    fitness_cromossomos = np.array([calcular_fitness(crom)
                                    for crom in cromossomos])

    melhor_fitness = max(fitness_cromossomos)
    melhor_cromossomo = cromossomos[np.argmax(fitness_cromossomos)]

    for gen in range(N_GERACOES):
        cromossomos = crossover(cromossomos, fitness_cromossomos)
        mutacao(cromossomos)
        fitness_cromossomos = np.array([calcular_fitness(crom)
                                        for crom in cromossomos])
        print("Geração", gen+1)

        if max(fitness_cromossomos) > melhor_fitness:
            melhor_fitness = max(fitness_cromossomos)
            melhor_cromossomo = cromossomos[np.argmax(fitness_cromossomos)]

        print("Melhor fitness:", melhor_fitness)

    print("-----------------------------------\nBalanço inicial: R$ 0.00")

    calcular_fitness(melhor_cromossomo, registrar_acoes=True)
    mostrar_solucao()


if __name__ == "__main__":
    main()
