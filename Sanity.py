import pandas as pd
import numpy as np

url_actual = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
df_actual = pd.read_csv(url_actual)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

column_mapping = {
    'Date': 'MatchDate',
    'HomeTeam': 'HomeTeam',
    'AwayTeam': 'AwayTeam',
    'FTHG': 'FullTimeHomeGoals',
    'FTAG': 'FullTimeAwayGoals',
    'FTR': 'FullTimeResult',
    'HS': 'HomeShots',
    'AS': 'AwayShots',
    'HST': 'HomeShotsOnTarget',
    'AST': 'AwayShotsOnTarget',
    'HC': 'HomeCorners',
    'AC': 'AwayCorners'
}

df_actual = df_actual.rename(columns=column_mapping)
df_actual['MatchDate'] = pd.to_datetime(df_actual['MatchDate'], format='%d/%m/%Y', errors='coerce')
df_actual = df_actual.dropna(subset=['FullTimeResult']).copy()

df_historico = pd.read_csv("epl_final.csv")
df_historico['MatchDate'] = pd.to_datetime(df_historico['MatchDate'], format='%Y-%m-%d')

df = pd.concat([df_historico, df_actual], ignore_index=True)
df = df.sort_values('MatchDate').reset_index(drop=True)
df = df.drop_duplicates(subset=['MatchDate', 'HomeTeam', 'AwayTeam'], keep='last')

target_map = {'H': 0, 'D': 1, 'A': 2}
df['target'] = df['FullTimeResult'].map(target_map)

print(f"Total de partidos cargados: {len(df)}")
print(f"Distribución del target:\n{df['FullTimeResult'].value_counts(normalize=True)}")

def calculate_features(df, window=5):
    home_df = df[['MatchDate', 'HomeTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals', 
                  'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget']].copy()
    home_df.columns = ['MatchDate', 'Equipo', 'GF', 'GA', 'TF', 'TA', 'TAF', 'TAA']
    home_df['es_local'] = 1

    away_df = df[['MatchDate', 'AwayTeam', 'FullTimeAwayGoals', 'FullTimeHomeGoals', 
                  'AwayShots', 'HomeShots', 'AwayShotsOnTarget', 'HomeShotsOnTarget']].copy()
    away_df.columns = ['MatchDate', 'Equipo', 'GF', 'GA', 'TF', 'TA', 'TAF', 'TAA']
    away_df['es_local'] = 0

    team_stats = pd.concat([home_df, away_df]).sort_values(['Equipo', 'MatchDate']).reset_index(drop=True)

    cols_a_promediar = ['GF', 'GA', 'TF', 'TA', 'TAF', 'TAA']
    
    for col in cols_a_promediar:
        team_stats[f'prom_{col}_general'] = team_stats.groupby('Equipo')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        team_stats[f'prom_{col}_condicion'] = team_stats.groupby(['Equipo', 'es_local'])[col].transform(lambda x: x.shift(0).rolling(window, min_periods=1).mean())

    team_stats['precision_tiro_general'] = team_stats['prom_TAF_general'] / (team_stats['prom_TF_general'] + 1e-5)

    feature_cols = [
        'MatchDate', 'Equipo',
        'prom_GF_general', 'prom_GA_general', 'prom_TF_general', 'prom_TA_general', 'prom_TAF_general', 'prom_TAA_general', 'precision_tiro_general',
        'prom_GF_condicion', 'prom_GA_condicion', 'prom_TF_condicion', 'prom_TA_condicion', 'prom_TAF_condicion', 'prom_TAA_condicion'
    ]

    df_features = pd.merge(
        df,
        team_stats[feature_cols],
        left_on=['MatchDate', 'HomeTeam'], right_on=['MatchDate', 'Equipo'], how='left'
    ).rename(columns={
        'prom_GF_general': 'local_prom_goles_favor_general',
        'prom_GA_general': 'local_prom_goles_contra_general',
        'prom_TF_general': 'local_prom_tiros_favor_general',
        'prom_TA_general': 'local_prom_tiros_contra_general',
        'prom_TAF_general': 'local_prom_tiros_arco_favor_general',
        'prom_TAA_general': 'local_prom_tiros_arco_contra_general',
        'precision_tiro_general': 'local_precision_tiro_general',
        'prom_GF_condicion': 'local_prom_goles_favor_como_local',
        'prom_GA_condicion': 'local_prom_goles_contra_como_local',
        'prom_TF_condicion': 'local_prom_tiros_favor_como_local',
        'prom_TA_condicion': 'local_prom_tiros_contra_como_local',
        'prom_TAF_condicion': 'local_prom_tiros_arco_favor_como_local',
        'prom_TAA_condicion': 'local_prom_tiros_arco_contra_como_local'
    }).drop(columns=['Equipo'])

    df_features = pd.merge(
        df_features,
        team_stats[feature_cols],
        left_on=['MatchDate', 'AwayTeam'], right_on=['MatchDate', 'Equipo'], how='left'
    ).rename(columns={
        'prom_GF_general': 'visitante_prom_goles_favor_general',
        'prom_GA_general': 'visitante_prom_goles_contra_general',
        'prom_TF_general': 'visitante_prom_tiros_favor_general',
        'prom_TA_general': 'visitante_prom_tiros_contra_general',
        'prom_TAF_general': 'visitante_prom_tiros_arco_favor_general',
        'prom_TAA_general': 'visitante_prom_tiros_arco_contra_general',
        'precision_tiro_general': 'visitante_precision_tiro_general',
        'prom_GF_condicion': 'visitante_prom_goles_favor_como_visitante',
        'prom_GA_condicion': 'visitante_prom_goles_contra_como_visitante',
        'prom_TF_condicion': 'visitante_prom_tiros_favor_como_visitante',
        'prom_TA_condicion': 'visitante_prom_tiros_contra_como_visitante',
        'prom_TAF_condicion': 'visitante_prom_tiros_arco_favor_como_visitante',
        'prom_TAA_condicion': 'visitante_prom_tiros_arco_contra_como_visitante'
    }).drop(columns=['Equipo'])

    return df_features

df_features = calculate_features(df, window=5)

columnas_verificacion = [
    'MatchDate', 'HomeTeam', 'AwayTeam', 
    'local_prom_goles_favor_general', 'visitante_prom_goles_favor_general',
    'local_prom_tiros_arco_favor_general', 'visitante_prom_tiros_arco_favor_general',
    'target'
]

df_features[columnas_verificacion].tail(10).to_html('vista_partidos.html')