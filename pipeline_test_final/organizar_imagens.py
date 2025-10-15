import os
import pandas as pd
import shutil

# --- CONFIGURAÇÃO ---
PASTA_ORIGEM_IMAGENS = 'PODI-LPR-01'
ARQUIVOS_CSV = {
    'fast': 'resultFastALPR.csv',
    'ultimate': 'resultUltimateALPR.csv',
    'tesseract': 'resultTesseractOCR.csv', 
    'paddle': 'resultPaddleOCR.csv'
}
COLUNA_CORRECAO = 'q_char_corrected'
PASTAS_DESTINO = {
    'common_error': 'common_error',
    'ultimate_errors': 'ultimateALPR_erros',
    'fast_errors': 'fastALPR_error',
    'common_success': 'common_images'
}
# --- FIM DA CONFIGURAÇÃO ---

def copiar_imagens(lista_imagens, pasta_destino):
    """
    Copia uma lista de imagens da pasta de origem para a pasta de destino.
    """
    print(f"\nCopiando {len(lista_imagens)} imagens para a pasta '{pasta_destino}'...")
    if not lista_imagens:
        print("Nenhuma imagem para copiar.")
        return

    for nome_imagem in lista_imagens:
        caminho_origem = os.path.join(PASTA_ORIGEM_IMAGENS, nome_imagem)
        caminho_destino = os.path.join(pasta_destino, nome_imagem)

        if os.path.exists(caminho_origem):
            shutil.copy2(caminho_origem, caminho_destino)
        else:
            print(f"  [Aviso] Imagem não encontrada na pasta de origem: {nome_imagem}")
    print(f"Cópia para '{pasta_destino}' concluída.")


def main():
    """
    Função principal que executa todo o processo.
    """
    print("Iniciando o processo de organização de imagens...")

    # 1. Criar as pastas de destino se não existirem
    for pasta in PASTAS_DESTINO.values():
        os.makedirs(pasta, exist_ok=True)
    print("Pastas de destino criadas com sucesso.")

    # 2. Carregar e preparar os DataFrames
    dataframes = {}
    try:
        for modelo, caminho in ARQUIVOS_CSV.items():
            df = pd.read_csv(caminho)
            df[f'{modelo}_correct'] = df[COLUNA_CORRECAO].isin([6, 7])
            dataframes[modelo] = df[['image', f'{modelo}_correct']].set_index('image')
    except FileNotFoundError as e:
        print(f"\n[ERRO] Arquivo não encontrado: {e.filename}")
        print("Por favor, verifique os caminhos na configuração 'ARQUIVOS_CSV'.")
        return
    except KeyError as e:
        print(f"\n[ERRO] A coluna {e} não foi encontrada em um dos arquivos CSV.")
        print(f"Verifique se o nome da coluna em 'COLUNA_CORRECAO' está correto.")
        return

    # 3. Juntar todos os resultados em um único DataFrame
    merged_df = pd.concat(dataframes.values(), axis=1, join='outer')
    merged_df.fillna(False, inplace=True)

    # 4. Aplicar as regras, obter as listas de imagens e gerar relatórios
    print("Analisando os resultados e aplicando as regras...")

    # Regra 1: Erro em comum (todos os modelos erraram)
    cond_common_error = (
        (merged_df['fast_correct'] == False) &
        (merged_df['ultimate_correct'] == False) &
        (merged_df['tesseract_correct'] == False) &
        (merged_df['paddle_correct'] == False)
    )
    imgs_common_error = merged_df[cond_common_error].index.tolist()

    # Regra 2: Erros do UltimateALPR (Ultimate errou E pelo menos um outro acertou)
    cond_ultimate_errors = (
        (merged_df['ultimate_correct'] == False) &
        ((merged_df['fast_correct'] == True) | (merged_df['tesseract_correct'] == True) | (merged_df['paddle_correct'] == True))
    )
    imgs_ultimate_errors = merged_df[cond_ultimate_errors].index.tolist()

    if imgs_ultimate_errors:
        df_report = merged_df[cond_ultimate_errors].copy()
        outros_modelos = ['fast_correct', 'tesseract_correct', 'paddle_correct']
        
        # Itera em cada linha para construir a string de modelos que acertaram
        s_acertos = df_report.apply(
            lambda row: ', '.join([
                model.replace('_correct', '') for model in outros_modelos if row[model]
            ]),
            axis=1
        )

        report_df = pd.DataFrame({
            'imagem': df_report.index,
            'modelo_com_erro': 'ultimate',
            'modelos_que_acertaram': s_acertos
        })
        
        caminho_csv = os.path.join(PASTAS_DESTINO['ultimate_errors'], 'relatorio_acertos.csv')
        report_df.to_csv(caminho_csv, index=False)
        print(f"-> Relatório de acertos para UltimateALPR salvo em: {caminho_csv}")

    # Regra 3: Erros do FastALPR (Fast errou E pelo menos um outro acertou)
    cond_fast_errors = (
        (merged_df['fast_correct'] == False) &
        ((merged_df['ultimate_correct'] == True) | (merged_df['tesseract_correct'] == True) | (merged_df['paddle_correct'] == True))
    )
    imgs_fast_errors = merged_df[cond_fast_errors].index.tolist()
    
    if imgs_fast_errors:
        df_report = merged_df[cond_fast_errors].copy()
        outros_modelos = ['ultimate_correct', 'tesseract_correct', 'paddle_correct']
        
        # Itera em cada linha para construir a string de modelos que acertaram
        s_acertos = df_report.apply(
            lambda row: ', '.join([
                model.replace('_correct', '') for model in outros_modelos if row[model]
            ]),
            axis=1
        )

        report_df = pd.DataFrame({
            'imagem': df_report.index,
            'modelo_com_erro': 'fast',
            'modelos_que_acertaram': s_acertos
        })
        
        caminho_csv = os.path.join(PASTAS_DESTINO['fast_errors'], 'relatorio_acertos.csv')
        report_df.to_csv(caminho_csv, index=False)
        print(f"-> Relatório de acertos para FastALPR salvo em: {caminho_csv}")

    # Regra 4: Acerto em comum (todos os modelos acertaram)
    cond_common_success = (
        (merged_df['fast_correct'] == True) &
        (merged_df['ultimate_correct'] == True) &
        (merged_df['tesseract_correct'] == True) &
        (merged_df['paddle_correct'] == True)
    )
    imgs_common_success = merged_df[cond_common_success].index.tolist()

    # 5. Copiar os arquivos de imagem para as pastas corretas
    copiar_imagens(imgs_common_error, PASTAS_DESTINO['common_error'])
    copiar_imagens(imgs_ultimate_errors, PASTAS_DESTINO['ultimate_errors'])
    copiar_imagens(imgs_fast_errors, PASTAS_DESTINO['fast_errors'])
    copiar_imagens(imgs_common_success, PASTAS_DESTINO['common_success'])

    print("\n-----------------------------------------")
    print("Processo finalizado com sucesso!")
    print(f"Resumo:")
    print(f"- {len(imgs_common_error)} imagens em '{PASTAS_DESTINO['common_error']}'")
    print(f"- {len(imgs_ultimate_errors)} imagens em '{PASTAS_DESTINO['ultimate_errors']}'")
    print(f"- {len(imgs_fast_errors)} imagens em '{PASTAS_DESTINO['fast_errors']}'")
    print(f"- {len(imgs_common_success)} imagens em '{PASTAS_DESTINO['common_success']}'")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()