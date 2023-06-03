import pandas as pd

# df = pd.read_csv('data_with_replies/Final/Result/autosave_jawa_province.csv')
df = pd.read_csv('indobert_smsa/testcase.csv')
print(df.shape)


# jakarta = df[df['Province'] == 'dki jakarta'].sample(n=150000)
# jawa_barat = df[df['Province'] == 'jawa barat'].sample(n=150000)
# jawa_timur = df[df['Province'] == 'jawa timur'].sample(n=150000)
# jawa_tengah = df[df['Province'] == 'jawa tengah'].sample(n=150000)

# testcase = pd.concat([jakarta, jawa_barat, jawa_timur, jawa_tengah])
# testcase.to_csv('indobert_smsa/testcase.csv', index=False)

