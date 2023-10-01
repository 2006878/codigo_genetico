import streamlit as st
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importando o arquivo params.py (ajuste o caminho conforme necessário)
from parametros import retorno_diario, emprestimo, investimento

# Definindo valores padrão (substitua pelos seus próprios valores ou pelos valores padrão)
N_GERACOES = 50
N_CROMOSSOMOS = 100
TOTAL_DIAS = 30
PROB_MUTACAO = 0.1

N_VARS_DIA = 2

NENHUMA = 0
EMPRESTIMO = 1
INVESTIMENTO = 2

# Conjunto de ações realizadas
informacoes_diarias = []

# Configurações da página Streamlit
st.set_page_config(page_title="Otimização de Caixa", layout="wide")

# Adicionando a logo da empresa
logo_path = "logo_zayon.jpeg"  # Substitua pelo caminho real para a sua logo
logo = Image.open(logo_path)

# Exibindo a logo no Streamlit
st.image(logo, use_column_width=False, width=250)

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
    st.write("### Resultado Final:")
    for dia, info_dia in enumerate(informacoes_diarias):
        st.write(f"**Dia {dia+1}:**")
        if info_dia['ação'] == EMPRESTIMO:
            n_dias = emprestimo[info_dia['tipo']]['dias']
            juros = emprestimo[info_dia['tipo']]['juros']
            st.write("- Ação realizada: Empréstimo.")
            st.write(f"- {juros:.1f}% de juros por {n_dias} dias.")
        elif info_dia['ação'] == INVESTIMENTO:
            n_dias = investimento[info_dia['tipo']]['dias']
            juros = investimento[info_dia['tipo']]['juros']
            st.write("- Ação realizada: Investimento.")
            st.write(f"- {juros:.1f}% de juros por {n_dias} dias.")
        else:
            st.write("- Nenhuma ação realizada.")
        st.write(f"- Entrada do caixa: R$ {info_dia['entrada']:.2f}")
        st.write(f"- Saída do caixa: R$ {info_dia['saida']:.2f}")
        balanco = info_dia['entrada'] - info_dia['saida']
        st.write(f"- Balanço no final do dia: R$ {balanco:.2f}")
        st.write("----")

def plot_fitness_evolution(fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o', linestyle='-', color='b')
    plt.title('Evolução do Fitness ao Longo das Gerações')
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    st.pyplot(plt)

# Interface do Streamlit
st.title("Otimização de caixa usando Algoritmo Genético")

# Elementos de entrada para os parâmetros
N_GERACOES = st.number_input("Número de Gerações", value=N_GERACOES)
N_CROMOSSOMOS = st.number_input("Número de Cromossomos", value=N_CROMOSSOMOS)
TOTAL_DIAS = st.number_input("Total de Dias", value=TOTAL_DIAS)
PROB_MUTACAO = st.number_input("Probabilidade de Mutação", value=PROB_MUTACAO)

# Botão para executar a análise
if st.button("Executar Análise"):
    informacoes_diarias = []
    cromossomos = gerar_solucoes_iniciais()
    fitness_cromossomos = np.array([calcular_fitness(crom)
                                    for crom in cromossomos])

    melhor_fitness = max(fitness_cromossomos)
    melhor_cromossomo = cromossomos[np.argmax(fitness_cromossomos)]

    fitness_history = [melhor_fitness]

    for gen in range(N_GERACOES):
        cromossomos = crossover(cromossomos, fitness_cromossomos)
        mutacao(cromossomos)
        fitness_cromossomos = np.array([calcular_fitness(crom)
                                        for crom in cromossomos])
        # st.write("Geração", gen+1)

        if max(fitness_cromossomos) > melhor_fitness:
            melhor_fitness = max(fitness_cromossomos)
            melhor_cromossomo = cromossomos[np.argmax(fitness_cromossomos)]

        fitness_history.append(melhor_fitness)

        # st.write("Melhor fitness:", melhor_fitness)

    plot_fitness_evolution(fitness_history)

    st.write("-----------------------------------\nBalanço inicial: R$ 0.00")

    calcular_fitness(melhor_cromossomo, registrar_acoes=True)
    mostrar_solucao()
