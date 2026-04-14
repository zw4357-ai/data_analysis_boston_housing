import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression


df=pd.read_csv("data/bostonHousingDataset.csv")

all_column= df.columns.tolist()

outcome = all_column[10]
predictors=all_column[:10]


r2_result=[]


print()
print("The R-squared score of each predictor to the outcome:")

for predictor in predictors:
    x = df[[predictor]].values
    y = df[outcome].values
    
    model = LinearRegression()
    model.fit(x, y)
    
    
    x_sorted = np.sort(x, axis=0)
    y_pred = model.predict(x_sorted)
 
    r2 = model.score(x, y)
    r2_result.append(r2)
    
    
    print(f"{predictor:20} : {r2:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x_sorted, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel(predictor.replace('_', ' ').title())
    plt.ylabel('House Value')
    plt.title(f'{predictor} vs House Value (R² = {r2:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(f'graph output/{predictor}_vs_house_value.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    

max_r2=max(r2_result)
max_index=r2_result.index(max_r2)
best_predictor=predictors[max_index]
print()
print(best_predictor+" predictor predicts the house price best.\n\n")


plt.figure(figsize=(10,6))
bars=plt.bar(predictors, r2_result)
bars[max_index].set_color('red')

plt.xlabel('Predictors')
plt.ylabel('R-squared Scores')
plt.title('R-squared Scores for All Predictors')
plt.xticks(rotation=45)

plt.savefig('graph output/r2_scores_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

