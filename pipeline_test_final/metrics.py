import pandas as pd
import matplotlib.pyplot as plt


def corrige_placa(text):
    text = ''.join(c for c in str(text).upper() if c.isalnum())
    if len(text) < 7:
        return [text, text]
    def is_letra(ch): return ch.isalpha()
    def is_num(ch): return ch.isdigit()
    padrao_antigo = [is_letra, is_letra, is_letra, is_num, is_num, is_num, is_num]
    padrao_novo = [is_letra, is_letra, is_letra, is_num, is_letra, is_num, is_num]
    def aplica_padrao(padrao, chars):
        corr = []
        for i, (ch, f) in enumerate(zip(chars, padrao)):
            if f(ch):
                corr.append(ch)
            else:
                if f == is_letra:
                    mapa = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', 'Q':'O'}
                    corr.append(mapa.get(ch, ch))
                else:
                    mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)
    placa_antiga = aplica_padrao(padrao_antigo, text)
    placa_nova = aplica_padrao(padrao_novo, text)
    return [placa_antiga, placa_nova]

def normalize(text):
    return ''.join(c for c in str(text).upper() if c.isalnum())

def char_matches(gold, pred):
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    return sum(a == b for a, b in zip(gold_norm, pred_norm))

def compare_columns(df, debug=False):
    df['gold_plate_norm'] = df['gold_plate'].apply(normalize)
    # Aplica correção nos textos dos modelos
    df['paddle_corr'] = df['paddle_text'].apply(lambda x: corrige_placa(x)[0])
    df['tesseract_corr'] = df['tesseract_text'].apply(lambda x: corrige_placa(x)[0])
    df['qwen_corr'] = df['qwen_text'].apply(lambda x: corrige_placa(x)[0])
    # Normaliza os textos corrigidos
    df['paddle_text_norm'] = df['paddle_corr'].apply(normalize)
    df['tesseract_text_norm'] = df['tesseract_corr'].apply(normalize)
    df['qwen_text_norm'] = df['qwen_corr'].apply(normalize)
    # Compara com o ground truth
    df['paddle_match'] = (df['gold_plate_norm'] == df['paddle_text_norm']).astype(int)
    df['tesseract_match'] = (df['gold_plate_norm'] == df['tesseract_text_norm']).astype(int)
    df['qwen_match'] = (df['gold_plate_norm'] == df['qwen_text_norm']).astype(int)
    df['paddle_char_matches'] = df.apply(lambda row: char_matches(row['gold_plate'], row['paddle_corr']), axis=1)
    df['tesseract_char_matches'] = df.apply(lambda row: char_matches(row['gold_plate'], row['tesseract_corr']), axis=1)
    df['qwen_char_matches'] = df.apply(lambda row: char_matches(row['gold_plate'], row['qwen_corr']), axis=1)
    if debug:
        df.to_csv('results/ocr_comparisons_debug.csv', index=False)
        print("Arquivo de debug salvo em results/ocr_comparisons_debug.csv")
    return df

def plot_percentage_and_absolute(df):
    total = len(df)
    acertos = [
        df['paddle_match'].sum(),
        df['tesseract_match'].sum(),
        df['qwen_match'].sum()
    ]
    erros = [total - a for a in acertos]
    porcentagem = [a / total * 100 for a in acertos]
    labels = ['Paddle', 'Tesseract', 'Qwen']

    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, porcentagem, color=['royalblue', 'orange', 'green'])
    plt.ylabel('Porcentagem de acertos (%)')
    plt.title(f'Acertos absolutos e porcentagem por modelo (Total de placas: {total})')

    for bar, acc, pct, err in zip(bars, acertos, porcentagem, erros):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{acc} ({pct:.1f}%)\nErros: {err}', 
                 ha='center', va='bottom', fontsize=12)

    plt.ylim(0, 110)
    plt.show()

def plot_char_match_table(df):
    table = pd.DataFrame({
        'Paddle': df['paddle_char_matches'].value_counts().sort_index(),
        'Tesseract': df['tesseract_char_matches'].value_counts().sort_index(),
        'Qwen': df['qwen_char_matches'].value_counts().sort_index()
    }).fillna(0).astype(int)
    print('\nDistribuição de acertos por caractere:')
    print(table)
    table.to_csv('results/char_match_table.csv')
    idx = [5, 6, 7]
    plt.figure(figsize=(8,6))
    for model, color in zip(['Paddle', 'Tesseract', 'Qwen'], ['royalblue', 'orange', 'green']):
        xvals = [i + 0.2 * ['Paddle', 'Tesseract', 'Qwen'].index(model) for i in idx]
        yvals = [table.at[i, model] if i in table.index else 0 for i in idx]
        bars = plt.bar(xvals, yvals, width=0.18, label=model, color=color)
        for x, y in zip(xvals, yvals):
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=12)
            
    plt.xticks(idx)
    plt.xlabel('Quantidade de caracteres corretos')
    plt.ylabel('Número de placas')
    plt.title('Distribuição de acertos por caractere (5, 6, 7), total de 300 inferências')
    plt.legend()
    plt.show()

def plot_top_errors(df, model_col, text_col, top_n=10):
    erro_df = df[df[model_col] == 0]
    erro_counts = erro_df['gold_plate'].value_counts().head(top_n)
    erro_texts = [erro_df[erro_df['gold_plate'] == plate][text_col].iloc[0] for plate in erro_counts.index]
    plt.figure(figsize=(10,6))
    plt.barh(erro_counts.index, erro_counts.values, color='crimson')
    plt.xlabel('Quantidade de erros')
    plt.title(f'Top {top_n} placas que o modelo {model_col.split("_")[0].capitalize()} mais errou')
    for i, (plate, count, pred) in enumerate(zip(erro_counts.index, erro_counts.values, erro_texts)):
        plt.text(count, i, f'Pred: {pred}', va='center', fontsize=10)
    plt.gca().invert_yaxis()
    plt.show()

def plot_time_comparison(df):
    labels = ['Paddle', 'Tesseract', 'Qwen']
    times = [
        df['paddle_time_ms'].mean(),
        df['tesseract_time_ms'].mean(),
        df['qwen_time_ms'].mean()
    ]
    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, times, color=['royalblue', 'orange', 'green'])
    plt.ylabel('Tempo médio (ms)')
    plt.title('Tempo médio de processamento por modelo')
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{t:.1f} ms', 
                 ha='center', va='bottom', fontsize=12)
    plt.show()

def plot_unique_errors(df):
    paddle_placas_erradas = set(df[df['paddle_match'] == 0]['gold_plate'])
    tesseract_placas_erradas = set(df[df['tesseract_match'] == 0]['gold_plate'])
    qwen_placas_erradas = set(df[df['qwen_match'] == 0]['gold_plate'])
    labels = ['Paddle', 'Tesseract', 'Qwen']
    valores = [len(paddle_placas_erradas), len(tesseract_placas_erradas), len(qwen_placas_erradas)]
    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, valores, color=['royalblue', 'orange', 'green'])
    plt.ylabel('Placas únicas erradas')
    plt.title('Quantidade de placas únicas erradas por modelo')
    for bar, val in zip(bars, valores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val), ha='center', va='bottom', fontsize=12)
    plt.show()

def plot_unique_plate_accuracy(df):
    unique_df = df.drop_duplicates(subset=['gold_plate'])
    total = len(unique_df)

    paddle_acertos = ((unique_df['paddle_char_matches'] >= 6)).sum()
    tesseract_acertos = ((unique_df['tesseract_char_matches'] >= 6)).sum()
    qwen_acertos = ((unique_df['qwen_char_matches'] >= 6)).sum()
    porcentagem = [
        paddle_acertos / total * 100,
        tesseract_acertos / total * 100,
        qwen_acertos / total * 100
    ]
    labels = ['Paddle', 'Tesseract', 'Qwen']
    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, porcentagem, color=['royalblue', 'orange', 'green'])
    plt.ylabel('Porcentagem de acertos (%)')
    plt.title(f'Porcentagem de acertos (6 ou 7 caracteres) por modelo\nPlacas únicas: {total}')
    for bar, acc, pct in zip(bars, [paddle_acertos, tesseract_acertos, qwen_acertos], porcentagem):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{acc} ({pct:.1f}%)', 
                 ha='center', va='bottom', fontsize=12)
    plt.ylim(0, 110)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('results/ocr_results_final_erodedilatedeskewupscale.csv')
    # df = pd.read_csv('results/ocr_results_sem_process_deskwuopscale_final.csv')
    # df = pd.read_csv('results/ocr_results_sem_process_final.csv')
    df = compare_columns(df, debug=True)
    # plot_percentage_and_absolute(df)
    plot_char_match_table(df)
    plot_top_errors(df, 'paddle_match', 'paddle_text', top_n=10)
    plot_top_errors(df, 'qwen_match', 'qwen_text', top_n=10)
    plot_time_comparison(df)
    plot_unique_plate_accuracy(df)