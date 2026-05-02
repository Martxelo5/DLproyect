import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath="data/insurance.csv"):
    """
    Performs Exploratory Data Analysis on the insurance dataset.
    Generates and saves visual plots to understand feature relationships.
    """
    print("---0: Exploratory Data Analysis (EDA) ---")
    
    # Check if data exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}. Please ensure the data folder exists.")
        return

    # We read the csv
    df = pd.read_csv(filepath)
    
    print("\nDataset Overview:")
    print(df.head())
    print("\nDataset Statistics:")
    print(df.describe())

    # We create a directory to save the plots
    os.makedirs("deliverable1/plots", exist_ok=True)

    # Plot 1: Distribution of Charges
    # We made a distribution of the insurance charges to see what charges are the most common ones
    plt.figure(figsize=(8, 5))
    sns.histplot(df['charges'], kde=True, bins=50, color='blue')
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges ($)')
    plt.ylabel('Frequency')
    plt.savefig('deliverable1/plots/charges_distribution.png')
    # plt.show()
    plt.close()
    print("- Saved charges_distribution.png")

    """ 
    With this plot we can see that most of the charges are bellow 20000$ 
    and then the trend goes down from that point onwards. 
    """

    # Plot 2: Correlation heatmap
    df_corr = df.copy()
    df_corr['smoker'] = df_corr['smoker'].map({'yes': 1, 'no': 0})
    df_corr['sex'] = df_corr['sex'].map({'male': 1, 'female': 0})

    plt.figure(figsize=(8, 6))
    corr_matrix = df_corr[['age', 'bmi', 'children', 'smoker', 'sex', 'charges',]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=14)
    plt.savefig('deliverable1/plots/correlation_heatmap.png')
    plt.close()
    print("- Saved correlation_heatmap.png")

    """ 
    After this plot we can see that the columns/atributes that are more meaningful 
    or more correlated with the charges are if the person is a smoker or not, the age and bmi. 
    With this information we will make some plots/visualizations to understand better the data 
    and see if we can see some trends in the data.
    """

    # Plot 3: Age vs charges scatter plot (colored by smoking status)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, palette='Set1', alpha=0.7)
    plt.title('Age vs. Charges (Colored by Smoking Status)', fontsize=15)
    plt.xlabel('Age')
    plt.ylabel('Charges ($)')
    plt.savefig('deliverable1/plots/Age_vs_charges.png')
    plt.close()
    print("- Saved Age_vs_charges.png")

    """
    This scatter plot of the age vs charges colored by the smoker status reveals 
    a "clear" trend. The charges of the insurance goes up at the same time the age goes up. 
    Moreover, when the person smokes, it happends the same to the trend,however the points 
    start from a higher position (higher charge). Nontheless, there is also a medium space 
    where there is a mix of smokers and non smokers.
    """

    # Plot 4: BMI vs Charges separated by Smoker Status
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', size='smoker', sizes=(50, 50), data=df, palette='Set2')
    plt.title('BMI vs. Charges (Colored by Smoking Status)', fontsize=14)
    plt.xlabel('Body Mass Index (BMI)')
    plt.ylabel('Charges ($)')
    plt.savefig('deliverable1/plots/bmi_smoker_interaction.png')
    plt.close()
    print("- Saved bmi_smoker_interaction.png")

    """
    Also, in this scatterplot we can see that the bmi does not have a clear impact 
    in the charge amount if the person is not a smoker.But we can interpret that 
    if the person smokes, the higher the bmi is the higher it's charge is going to be.
    """

    # Plot 5: Insurance charges by region
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region', y='charges', hue='smoker', data=df, palette='Set1')
    plt.title('Insurance Charges by Region and Smoking Status', fontsize=15)
    plt.xlabel('Region')
    plt.ylabel('Charges ($)')
    plt.legend(title='Smoker?')
    plt.savefig('deliverable1/plots/Region_smoker_distribution.png')
    plt.close()
    print("- Saved Region_smoker_distribution.png")

    """
    Finally, in this boxplot we can observe if there is a diferrence in charges between regions. 
    We can clearly see that if the person is not a smoker the region has not a big impact on the charges.
    Instead, if the person smokes, we can observe that the region has a bigger impact. 
    The regions of southwest and southest have a quite bigger median of charges compared to northwest and northeast.
    """

    print("EDA Complete! All plots have been saved to the 'plots/' directory.\n")