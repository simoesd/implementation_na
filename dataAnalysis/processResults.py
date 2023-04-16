import pandas as pd
from scipy.stats import mannwhitneyu
import sys
import openpyxl

pd.set_option('display.max_rows', None)


############# Setup
inputFile = sys.argv[1];
df = pd.read_csv(inputFile, sep=',', on_bad_lines='skip', index_col=False)
baseColumns = ['R', 'Mutation Algorithm', 'Problem', 'Input Dim', 'Hidden Dim', 'Output Dim', 'Iteration',
       'Generation', 'Score', 'Solution']
df.dropna(inplace=True, subset=baseColumns)
df.drop(['Iteration', 'Solution', 'Input Dim', 'Hidden Dim'], axis=1, inplace=True);

################## MEDIAN
# Calculates the median generation and score for each problem/algorithm combination found.
groupedDF = df.groupby(axis=0, by=["R", "Problem", "Output Dim", "Mutation Algorithm"])
medianDF = groupedDF.median()

################### MANN-WHITNEY U
# Merge the three columns that identify the problem into one, for easier grouping
# Moves the created column to the left of the dataframe
df['Problem Parameters'] = [", ".join([str(y) for y in x]) for x in df[["R", "Problem", "Output Dim"]].values.tolist()]
df.set_index('Problem Parameters')
df.drop(["R", "Problem", "Output Dim"], axis=1, inplace=True);
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

mannWhitneyUDF = pd.DataFrame(columns=['Problem', 'Mutation Algorithm 1', 'Mutation Algorithm 2', 'Generation U Statistic', 'Generation P Value'])
# For each unique problem found in the input sheet, executes a Mann-Whitney U significance test on the generation column between all found algorithms
problemNames = df['Problem Parameters'].unique()
for problemName in problemNames:
    problemResults = df.loc[df['Problem Parameters'] == problemName]
    mutationNames = problemResults['Mutation Algorithm'].unique()
    for x in mutationNames:
       for y in mutationNames:
              if x != y:
                     mannWhitneyUGen = mannwhitneyu(problemResults.loc[df['Mutation Algorithm'] == x]['Generation'], problemResults.loc[df['Mutation Algorithm'] == y]['Generation'])
                     mannWhitneyUDF.loc[len(mannWhitneyUDF)] = [
                            problemName,
                            x,
                            y,
                            mannWhitneyUGen.statistic,
                            mannWhitneyUGen.pvalue,
                     ]

# Write results to output file, one sheet for the median results and one for significance tests
exportFile = f'{inputFile}'
if inputFile.startswith('../'):
     exportFile = f'../processed{inputFile[3:]}'
writer = pd.ExcelWriter(exportFile, engine="openpyxl")
medianDF.to_excel(writer, sheet_name='Medians', index=False)
mannWhitneyUDF.to_excel(writer, sheet_name='Mann Whitney U', index=False)
writer.close()