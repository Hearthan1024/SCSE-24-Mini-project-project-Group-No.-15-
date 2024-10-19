import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

class DataInspection:
    def __init__(self):
        self.df = None  # store dataset

    def load_csv(self, file_path):
        """Load CSV and show the information of variables"""
        try:
            self.df = pd.read_csv(file_path)
            print("Following are the variables in your dataset:")
            self.show_variable_info()  
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            return

    def show_variable_info(self):
        """Showing the variable list"""
        if self.df is None:
            print("No dataset loaded.")
            return

        variable_info = []
        for col in self.df.columns:
            col_type = self._get_variable_type(self.df[col])
            
            # calculate mean/median/mode for numeric variables
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stat_value = f"{self.df[col].mean():.2f}"  # if numeric, show mean
                kurt = f"{self.df[col].kurtosis():.2f}"
                skew = f"{self.df[col].skew():.2f}"
            elif pd.api.types.is_bool_dtype(self.df[col]):
                stat_value = self.df[col].mode()[0]  
                kurt = skew = "NA"
            else:
                stat_value = self.df[col].mode()[0] if not self.df[col].mode().empty else "NA"
                kurt = skew = "NA"

            variable_info.append({
                "Variable": col,
                "Type": col_type,
                "Mean / Median / Mode": stat_value,
                "Kurtosis": kurt,
                "Skewness": skew
            })

        info_df = pd.DataFrame(variable_info)
        print(info_df.to_string(index=False))  
    def _get_variable_type(self, series):
        if pd.api.types.is_bool_dtype(series):
            return 'Nominal'  
        elif pd.api.types.is_numeric_dtype(series):
            return 'Ratio' if series.nunique() > 10 else 'Ordinal'
        else:
            return 'Nominal'


    def show_plot_variables(self):
        """List variables for plotting"""
        print("\nFollowing variables are available for plot distribution:")

        for idx, col in enumerate(self.df.columns, start=1):
            print(f"{idx}. {col}")

        print(f"{len(self.df.columns) + 1}. BACK")
        print(f"{len(self.df.columns) + 2}. QUIT")

    def plot_distribution(self, variable):
        """Choose the variable to plot and plot it"""
        if variable not in self.df.columns:
            print(f"Variable '{variable}' not found.")
            return

        var_type = self._get_variable_type(self.df[variable])
   
        if var_type == 'Ratio':
            # Ratio variables - use histogram or density plot
            num_obs = self.df[variable].dropna().shape[0]  # Available observations

            if num_obs <= 1000:
                # use histogram
                sns.histplot(self.df[variable], kde=False)
                plt.title(f"Histogram of {variable} (Ratio)")
            else:
                # use density plot
                sns.kdeplot(self.df[variable], fill=True)
                plt.title(f"Density Plot of {variable} (Ratio)")

            plt.xlabel(variable)
            plt.ylabel("Frequency")
            plt.show()

        elif var_type == 'Ordinal':
            # use bar chart
            sns.countplot(x=variable, data=self.df)
            plt.title(f"Bar Plot of {variable} (Ordinal)")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.show()

        elif var_type == 'Nominal':
            # use bar chart
            sns.countplot(x=variable, data=self.df)
            plt.title(f"Bar Plot of {variable} (Nominal/Boolean)")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.show()

        else:
            print(f"Variable '{variable}' has an unsupported data type for plotting.")

class DataAnalysis:
    def __init__(self, df):
        self.df = df

    def _get_variable_type(self, series):
        if pd.api.types.is_bool_dtype(series):
            return 'Nominal'  
        elif pd.api.types.is_numeric_dtype(series):
            return 'Ratio' if series.nunique() > 10 else 'Ordinal'
        else:
            return 'Nominal'
    def show_anova_variables(self):
        """List variables for ANOVA"""
        variable_info = [
            {"Variable": col, "Type": self._get_variable_type(self.df[col])}
            for col in self.df.columns
        ]
        info_df = pd.DataFrame(variable_info)
        print("\nFor ANOVA, following are the variables available:")
        print(info_df.to_string(index=False))
        
    def anova_analysis(self, continuous_var, categorical_var):
        """Do ANOVA or Kruskal-Wallis test"""
        sns.set(style="whitegrid")

        # Draw a Q-Q plot
        stats.probplot(self.df[continuous_var], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {continuous_var}")
        plt.show()
        # Perform normality test and select appropriate testing method
        def test_normality(data, size_limit=2000):
            """根据样本大小自动选择正态性检验方法"""
            if len(data) > size_limit:
                # Anderson Darling test returns AndersonResult object, does not return p-value
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]  # Select the critical value corresponding to the 5% significance level
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value  # If the statistic is less than the critical value, the data conforms to a normal distribution
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        # Obtain normality test results
        test_type, stat, p_value, is_normal, result = test_normality(self.df[continuous_var])

        # Print normality test results
        if test_type == 'Shapiro-Wilk':
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, p-value = {p_value:.15f}")
        else:
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, Critical Value = {result.critical_values[2]:.4f}")
            
        # Statement of null hypothesis: There is no significant difference in the mean values of continuous variables under different classifications
        print(f"\nNull Hypothesis: There is no significant difference in the mean of '{continuous_var}' "
               f"across the groups of '{categorical_var}'.")

        # Choose ANOVA or Kruskal Wallis test based on normality
        if is_normal:
            print(f"'{continuous_var}' is normally distributed. Performing ANOVA...")
            f_stat, p_value = stats.f_oneway(
                *[group[1][continuous_var] for group in self.df.groupby(categorical_var)]
            )
            print(f"\nANOVA Result: F-statistic = {f_stat}, p-value = {p_value}")
        else:
            print(f"'{continuous_var}' is not normally distributed, as shown in the Q-Q plot.")
            print("Performing Kruskal-Wallis Test instead...")
            stat, p_value = stats.kruskal(
                *[group[1][continuous_var] for group in self.df.groupby(categorical_var)]
            )
            print(f"\nKruskal-Wallis Result:\nStatistic = {stat}, p-value = {p_value}")

        # Interpretation results
        if p_value < 0.05:
            print("Result is statistically significant. Therefore, your Null Hypothesis is rejected.")
            print(f"There is a statistically significant difference in the average '{continuous_var}' "
                  f"across the categories of '{categorical_var}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")


    def show_variables(self, var_type):
        """Display variables of specified types"""
        variables = [col for col in self.df.columns if self._get_variable_type(self.df[col]) == var_type]
        var_df = pd.DataFrame({"Type": [var_type] * len(variables), "Variable": variables})
        print(var_df.to_string(index=False))
        return variables

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        """Perform t-test or Mann Whitney U test, generate charts and output conclusions"""
        # List categorical and continuous variables
        print(f"Selected continuous variable: {continuous_var}")
        print(f"Selected categorical variable: {categorical_var}")

        # Data grouped by categorical variables
        groups = [group[continuous_var].dropna() for _, group in self.df.groupby(categorical_var)]

        # Draw Q-Q diagram
        print(f"\nChecking normality for '{continuous_var}'...")
        sns.set(style="whitegrid")
        stats.probplot(self.df[continuous_var], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {continuous_var}")
        plt.show()

        # Automatically perform normality test
        def test_normality(data, size_limit=2000):
            """Automatically select normality testing method based on sample size"""
            if len(data) > size_limit:
                # Anderson-Darling
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value 
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        test_type, stat, p_value, is_normal, result = test_normality(self.df[continuous_var])

        if test_type == 'Shapiro-Wilk':
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, p-value = {p_value:.15f}")
        else:
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, Critical Value = {result.critical_values[2]:.4f}")

        # (Boxplot)
        sns.boxplot(x=categorical_var, y=continuous_var, data=self.df)
        plt.title(f"Boxplot of {continuous_var} by {categorical_var}")
        plt.show()

        #  t-test or Mann-Whitney U Test
        if is_normal:
            print(f"'{continuous_var}' is normally distributed. Performing t-Test...")
            stat, p_value = stats.ttest_ind(*groups)
            test_name = "t-Test"
        else:
            print(f"'{continuous_var}' is not normally distributed. Performing Mann-Whitney U Test...")
            stat, p_value = stats.mannwhitneyu(*groups)
            test_name = "Mann-Whitney U Test"

        print(f"\n{test_name} Result:")
        print(f"Statistic = {stat:.4f}, p-value = {p_value:.15f}")

        print(f"\nNull Hypothesis: There is no significant difference in the mean of '{continuous_var}' "
              f"across the groups of '{categorical_var}'.")

        if p_value < 0.05:
            print("The result is statistically significant. Therefore, your Null Hypothesis is rejected.")
            print(f"There is a significant difference in '{continuous_var}' across the groups of '{categorical_var}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")

    def show_categorical_variables(self):
        """Display all nominal and ordinal variables"""
        variables = [
            {"Variable": col, "Type": self._get_variable_type(self.df[col])}
            for col in self.df.columns
            if self._get_variable_type(self.df[col]) in ['Nominal', 'Ordinal']
        ]
        info_df = pd.DataFrame(variables)
        print("\nAvailable categorical variables for Chi-Square Test (Nominal/Ordinal):")
        print(info_df.to_string(index=False))
        return [var['Variable'] for var in variables]

    def chi_square_test(self, var1, var2):
        """Perform chi square test and return results, state the null hypothesis and whether to reject the null hypothesis"""
    
        print(f"\nSelected variables: '{var1}' (Categorical), '{var2}' (Categorical)")
    
        # (Contingency Table)
        print(f"\nGenerating Contingency Table for '{var1}' and '{var2}'...")
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        print("\nContingency Table:")
        print(contingency_table)

        # Check if there are enough categories in the contingency table for chi square test
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            print("One of the variables does not have enough categories for Chi-Square Test.")
            return

        # Perform chi square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # Visualize contingency table data (stacked bar chart)
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'Stacked Bar Chart of {var1} vs {var2}')
        plt.xlabel(var1)
        plt.ylabel("Count")
        plt.show()

        print(f"\nChi-Square Test Result:")
        print(f"Chi2 = {chi2:.4f}, p-value = {p:.15f}, Degrees of Freedom = {dof}")

        # Output expected frequency
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        print("\nExpected Frequencies (expected frequency):")
        print(expected_df)

        print(f"\nNull Hypothesis: There is no significant association between '{var1}' and '{var2}'.")

        if p < 0.05:
            print("The result is statistically significant. Therefore, the Null Hypothesis is rejected.")
            print(f"There is a significant association between '{var1}' and '{var2}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")
            print(f"There is no significant association between '{var1}' and '{var2}'.")


    def show_interval_variables(self):
        """(Interval)"""
        variables = [
            {"Variable": col, "Type": "Ratio"}  # Ratio
            for col in self.df.columns
            if self._get_variable_type(self.df[col]) == 'Ratio'
        ]
        if not variables:
            print("No continuous (Interval) variables available for Regression.")
            return []

        info_df = pd.DataFrame(variables)
        print("\nAvailable continuous (Interval) variables for Regression:")
        print(info_df.to_string(index=False))
        return [var["Variable"] for var in variables]


    def perform_regression(self, x_var, y_var):
        """Perform linear regression analysis, test normality, generate charts and output conclusions"""
    
        # List the data types of independent and dependent variables
        x_type = self._get_variable_type(self.df[x_var])
        y_type = self._get_variable_type(self.df[y_var])
        print(f"\nData Types: {x_var} (X - Independent): {x_type}, {y_var} (Y - Dependent): {y_type}")
    
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()

        # Ensure that the lengths of the two variables are consistent
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]

        def test_normality(data, size_limit=2000):
            """根据样本大小自动选择正态性检验方法"""
            if len(data) > size_limit:
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]  
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value 
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        # Perform normality tests on independent and dependent variables
        test_type_x, stat_x, p_value_x, is_normal_x, result_x = test_normality(X)
        test_type_y, stat_y, p_value_y, is_normal_y, result_y = test_normality(Y)

        # Print the normality test results of the independent and dependent variables
        print(f"\nNormality Test for {x_var} (X):")
        if test_type_x == 'Shapiro-Wilk':
            print(f"({test_type_x}) Statistic = {stat_x:.4f}, p-value = {p_value_x:.15f}")
        else:
            print(f"({test_type_x}) Statistic = {stat_x:.4f}, Critical Value = {result_x.critical_values[2]:.4f}")

        print(f"\nNormality Test for {y_var} (Y):")
        if test_type_y == 'Shapiro-Wilk':
            print(f"({test_type_y}) Statistic = {stat_y:.4f}, p-value = {p_value_y:.15f}")
        else:
            print(f"({test_type_y}) Statistic = {stat_y:.4f}, Critical Value = {result_y.critical_values[2]:.4f}")

        # State the null hypothesis of linear regression
        print(f"\nNull Hypothesis: There is no significant linear relationship between '{x_var}' and '{y_var}'.")

        # Check if the independent and dependent variables follow a normal distribution
        if not is_normal_x or not is_normal_y:
            print(f"One or both variables are not normally distributed. Proceed with caution when interpreting the results.")
    
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

        # Draw scatter plots and regression lines
        sns.regplot(x=X, y=Y, line_kws={'color': 'red'})
        plt.title(f'Regression: {y_var} ~ {x_var}')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

        print("\nRegression Analysis Results:")
        print(f"Slope: {slope:.4f}")
        print("Interpretation: The slope indicates the strength and direction of the relationship between the variables.")

        print(f"Intercept: {intercept:.4f}")
        print(f"Prediction: When the value of '{x_var}' is 0, the predicted value of '{y_var}' is {intercept:.4f}.")

        print(f"R-squared: {r_value**2:.4f}")
        print("Explanation: The R-squared value indicates the model's goodness of fit. "
              "A higher value means the model fits the data better.")

        print(f"P-value: {p_value:.15f}")
        # p-value
        if p_value < 0.05:
            print(f"The result is statistically significant. Therefore, the Null Hypothesis is rejected.")
            print(f"'{x_var}' has a significant impact on '{y_var}'.")
        else:
            print(f"The result is not statistically significant. The Null Hypothesis cannot be rejected.")
            print(f"'{x_var}' does not have a significant impact on '{y_var}'.")

        print(f"Standard Error: {std_err:.4f}")
        print("Explanation: The standard error estimates the uncertainty of the regression coefficient.")

class SentimentAnalysis:
    def __init__(self, df):
        self.df = df
        self.analyzer = SentimentIntensityAnalyzer()
        try:
            from transformers import pipeline
            self.distilbert_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        except ImportError:
            self.distilbert_pipeline = None

    def analyze(self, text_column, method):
        """Perform sentiment analysis"""
        print(f"Selected column for analysis: {text_column}")
        print(f"Data type: {self.df[text_column].dtype}")

        if method == '1':
            return self.vader_analysis(text_column)
        elif method == '2':
            return self.textblob_analysis(text_column)
        elif method == '3':
            if self.distilbert_pipeline:
                return self.distilbert_analysis(text_column)
            else:
                print("DistilBERT is not available. Please install transformers.")
                return None
        else:
            print("Invalid choice!")
            return None

    def vader_analysis(self, column):
        """VADER """
        results = []
        for text in self.df[column].fillna(""):
            score = self.analyzer.polarity_scores(text)['compound']
            sentiment = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
            results.append({"Text": text, "Score": score, "Sentiment": sentiment})
        result_df = pd.DataFrame(results)
        print("\nVADER Sentiment Analysis Results:")
        print(result_df)
        return result_df  

    def textblob_analysis(self, column):
        """TextBlob """
        results = []
        for text in self.df[column].fillna(""):
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
            results.append({"Text": text, "Polarity": polarity, "Sentiment": sentiment, "Subjectivity": subjectivity})
        result_df = pd.DataFrame(results)
        print("\nTextBlob Sentiment Analysis Results:")
        print(result_df)
        return result_df 

    def distilbert_analysis(self, column):
        """DistilBERT """
        results = []
        for text in self.df[column].fillna(""):
            result = self.distilbert_pipeline(text)[0]
            score = result['score']
            label = result['label']
            sentiment = 'Positive' if label in ['4 stars', '5 stars'] else 'Neutral' if label == '3 stars' else 'Negative'
            results.append({"Text": text, "Score": score, "Sentiment": sentiment})
        result_df = pd.DataFrame(results)
        print("\nDistilBERT Sentiment Analysis Results:")
        print(result_df)
        return result_df 

