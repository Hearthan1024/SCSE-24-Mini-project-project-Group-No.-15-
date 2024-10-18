import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# 数据加载与检查模块
class DataInspection:
    def __init__(self):
        self.df = None  # 存储数据集

    def load_csv(self, file_path):
        """加载 CSV 数据集并展示变量信息"""
        try:
            self.df = pd.read_csv(file_path)
            print("Following are the variables in your dataset:")
            self.show_variable_info()  # 加载成功后展示变量表
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
            return

    def show_variable_info(self):
        """展示变量信息表"""
        if self.df is None:
            print("No dataset loaded.")
            return

        variable_info = []
        for col in self.df.columns:
            col_type = self._get_variable_type(self.df[col])
            
            # 计算均值/中位数/众数
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stat_value = f"{self.df[col].mean():.2f}"  # 如果是数值型，显示均值
                kurt = f"{self.df[col].kurtosis():.2f}"
                skew = f"{self.df[col].skew():.2f}"
            elif pd.api.types.is_bool_dtype(self.df[col]):
                stat_value = self.df[col].mode()[0]  # 布尔类型显示众数
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
        print(info_df.to_string(index=False))  # 打印表格

    def _get_variable_type(self, series):
        if pd.api.types.is_bool_dtype(series):
            return 'Nominal'  # 布尔类型应视为名义变量
        elif pd.api.types.is_numeric_dtype(series):
            return 'Ratio' if series.nunique() > 10 else 'Ordinal'
        else:
            return 'Nominal'


    def show_plot_variables(self):
        """列出可用于绘图的变量，并添加选项编号"""
        print("\nFollowing variables are available for plot distribution:")

        # 构建变量列表并显示编号
        for idx, col in enumerate(self.df.columns, start=1):
            print(f"{idx}. {col}")

        print(f"{len(self.df.columns) + 1}. BACK")
        print(f"{len(self.df.columns) + 2}. QUIT")

    def plot_distribution(self, variable):
        """根据变量类型选择适当的图表并绘制"""
        if variable not in self.df.columns:
            print(f"Variable '{variable}' not found.")
            return

        var_type = self._get_variable_type(self.df[variable])
   
        if var_type == 'Ratio':
            # 连续变量 - 根据观测值数量选择直方图或密度图
            num_obs = self.df[variable].dropna().shape[0]  # 有效观测值数量

            if num_obs <= 1000:
                # 观测值少于 1000，使用直方图
                sns.histplot(self.df[variable], kde=False)
                plt.title(f"Histogram of {variable} (Ratio)")
            else:
                # 观测值多于 1000，使用密度图
                sns.kdeplot(self.df[variable], fill=True)
                plt.title(f"Density Plot of {variable} (Ratio)")

            plt.xlabel(variable)
            plt.ylabel("Frequency")
            plt.show()

        elif var_type == 'Ordinal':
            # 有序变量 - 使用柱状图
            sns.countplot(x=variable, data=self.df)
            plt.title(f"Bar Plot of {variable} (Ordinal)")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.show()

        elif var_type == 'Nominal':
            # 名义变量和布尔变量 - 使用柱状图
            sns.countplot(x=variable, data=self.df)
            plt.title(f"Bar Plot of {variable} (Nominal/Boolean)")
            plt.xlabel(variable)
            plt.ylabel("Count")
            plt.show()

        else:
            print(f"Variable '{variable}' has an unsupported data type for plotting.")

# 数据分析模块
class DataAnalysis:
    def __init__(self, df):
        self.df = df

    def _get_variable_type(self, series):
        if pd.api.types.is_bool_dtype(series):
            return 'Nominal'  # 布尔类型应视为名义变量
        elif pd.api.types.is_numeric_dtype(series):
            return 'Ratio' if series.nunique() > 10 else 'Ordinal'
        else:
            return 'Nominal'
    def show_anova_variables(self):
        """列出可用于 ANOVA 的变量"""
        variable_info = [
            {"Variable": col, "Type": self._get_variable_type(self.df[col])}
            for col in self.df.columns
        ]
        info_df = pd.DataFrame(variable_info)
        print("\nFor ANOVA, following are the variables available:")
        print(info_df.to_string(index=False))
        
    def anova_analysis(self, continuous_var, categorical_var):
        """执行 ANOVA 或 Kruskal-Wallis 检验，自动选择正态性检验"""
        sns.set(style="whitegrid")

        # 绘制 Q-Q 图
        stats.probplot(self.df[continuous_var], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {continuous_var}")
        plt.show()
        # 自动进行正态性检验，选择合适的检验方法
        def test_normality(data, size_limit=2000):
            """根据样本大小自动选择正态性检验方法"""
            if len(data) > size_limit:
                # Anderson-Darling 检验返回 AndersonResult 对象，不返回 p-value
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]  # 选择 5% 显著性水平对应的临界值
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value  # 如果统计量小于临界值，则数据符合正态分布
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        # 获取正态性检验结果
        test_type, stat, p_value, is_normal, result = test_normality(self.df[continuous_var])

        # 打印正态性检验结果
        if test_type == 'Shapiro-Wilk':
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, p-value = {p_value:.15f}")
        else:
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, Critical Value = {result.critical_values[2]:.4f}")
            
        # 陈述原假设：不同分类下，连续变量的平均值没有显著差异
        print(f"\nNull Hypothesis: There is no significant difference in the mean of '{continuous_var}' "
               f"across the groups of '{categorical_var}'.")

        # 根据正态性选择 ANOVA 或 Kruskal-Wallis 检验
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

        # 解读结果
        if p_value < 0.05:
            print("Result is statistically significant. Therefore, your Null Hypothesis is rejected.")
            print(f"There is a statistically significant difference in the average '{continuous_var}' "
                  f"across the categories of '{categorical_var}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")


    def show_variables(self, var_type):
        """展示指定类型的变量"""
        variables = [col for col in self.df.columns if self._get_variable_type(self.df[col]) == var_type]
        var_df = pd.DataFrame({"Type": [var_type] * len(variables), "Variable": variables})
        print(var_df.to_string(index=False))
        return variables

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        """执行 t-Test 或 Mann-Whitney U Test，并生成图表和输出结论"""
        # 列出分类变量和连续变量
        print(f"Selected continuous variable: {continuous_var}")
        print(f"Selected categorical variable: {categorical_var}")

        # 按分类变量分组后的数据
        groups = [group[continuous_var].dropna() for _, group in self.df.groupby(categorical_var)]

        # 绘制 Q-Q 图
        print(f"\nChecking normality for '{continuous_var}'...")
        sns.set(style="whitegrid")
        stats.probplot(self.df[continuous_var], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {continuous_var}")
        plt.show()

        # 自动进行正态性检验
        def test_normality(data, size_limit=2000):
            """根据样本大小自动选择正态性检验方法"""
            if len(data) > size_limit:
                # Anderson-Darling 检验
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]  # 选择 5% 显著性水平对应的临界值
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value  # 判断是否符合正态分布
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        # 获取正态性检验结果
        test_type, stat, p_value, is_normal, result = test_normality(self.df[continuous_var])

        # 打印正态性检验结果
        if test_type == 'Shapiro-Wilk':
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, p-value = {p_value:.15f}")
        else:
            print(f"Normality Test ({test_type}): Statistic = {stat:.4f}, Critical Value = {result.critical_values[2]:.4f}")

        # 绘制箱线图 (Boxplot)
        sns.boxplot(x=categorical_var, y=continuous_var, data=self.df)
        plt.title(f"Boxplot of {continuous_var} by {categorical_var}")
        plt.show()

        # 应用 t-test 或 Mann-Whitney U Test
        if is_normal:
            print(f"'{continuous_var}' is normally distributed. Performing t-Test...")
            stat, p_value = stats.ttest_ind(*groups)
            test_name = "t-Test"
        else:
            print(f"'{continuous_var}' is not normally distributed. Performing Mann-Whitney U Test...")
            stat, p_value = stats.mannwhitneyu(*groups)
            test_name = "Mann-Whitney U Test"

        # 打印统计检验结果
        print(f"\n{test_name} Result:")
        print(f"Statistic = {stat:.4f}, p-value = {p_value:.15f}")

        # 陈述原假设
        print(f"\nNull Hypothesis: There is no significant difference in the mean of '{continuous_var}' "
              f"across the groups of '{categorical_var}'.")

        # 判断是否拒绝原假设
        if p_value < 0.05:
            print("The result is statistically significant. Therefore, your Null Hypothesis is rejected.")
            print(f"There is a significant difference in '{continuous_var}' across the groups of '{categorical_var}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")

    def show_categorical_variables(self):
        """展示所有名义 (Nominal) 和有序 (Ordinal) 变量"""
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
        """执行卡方检验并返回结果，陈述原假设和是否拒绝原假设"""
    
        # 列出所选变量的类型
        print(f"\nSelected variables: '{var1}' (Categorical), '{var2}' (Categorical)")
    
        # 生成列联表 (Contingency Table)
        print(f"\nGenerating Contingency Table for '{var1}' and '{var2}'...")
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        print("\nContingency Table:")
        print(contingency_table)

        # 检查列联表中是否有足够的类别进行卡方检验
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            print("One of the variables does not have enough categories for Chi-Square Test.")
            return

        # 执行卡方检验
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

        # 可视化列联表数据 (堆叠柱状图)
        contingency_table.plot(kind='bar', stacked=True)
        plt.title(f'Stacked Bar Chart of {var1} vs {var2}')
        plt.xlabel(var1)
        plt.ylabel("Count")
        plt.show()

        print(f"\nChi-Square Test Result:")
        print(f"Chi2 = {chi2:.4f}, p-value = {p:.15f}, Degrees of Freedom = {dof}")

        # 输出期望频数
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        print("\nExpected Frequencies (期望频数):")
        print(expected_df)

        # 陈述原假设，明确用户选择的两个变量
        print(f"\nNull Hypothesis: There is no significant association between '{var1}' and '{var2}'.")

        # 判断是否拒绝原假设
        if p < 0.05:
            print("The result is statistically significant. Therefore, the Null Hypothesis is rejected.")
            print(f"There is a significant association between '{var1}' and '{var2}'.")
        else:
            print("The result is not statistically significant. Null hypothesis cannot be rejected.")
            print(f"There is no significant association between '{var1}' and '{var2}'.")


    def show_interval_variables(self):
        """展示所有连续变量 (Interval)"""
        variables = [
            {"Variable": col, "Type": "Ratio"}  # 连续变量被归类为 Ratio
            for col in self.df.columns
            if self._get_variable_type(self.df[col]) == 'Ratio'
        ]
        if not variables:
            print("No continuous (Interval) variables available for Regression.")
            return []

    # 展示可用的连续变量
        info_df = pd.DataFrame(variables)
        print("\nAvailable continuous (Interval) variables for Regression:")
        print(info_df.to_string(index=False))
        return [var["Variable"] for var in variables]


    def perform_regression(self, x_var, y_var):
        """执行线性回归分析，测试正态性，生成图表并输出结论"""
    
        # 列出自变量和因变量的数据类型
        x_type = self._get_variable_type(self.df[x_var])
        y_type = self._get_variable_type(self.df[y_var])
        print(f"\nData Types: {x_var} (X - Independent): {x_type}, {y_var} (Y - Dependent): {y_type}")
    
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()

        # 确保两个变量长度一致
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]

        # 自动进行正态性检验
        def test_normality(data, size_limit=2000):
            """根据样本大小自动选择正态性检验方法"""
            if len(data) > size_limit:
                result = stats.anderson(data)
                stat = result.statistic
                critical_value = result.critical_values[2]  # 选择 5% 显著性水平对应的临界值
                test_type = 'Anderson-Darling'
                is_normal = stat < critical_value  # 如果统计量小于临界值，则数据符合正态分布
                return test_type, stat, None, is_normal, result
            else:
                stat, p_value = stats.shapiro(data)
                test_type = 'Shapiro-Wilk'
                is_normal = p_value >= 0.05
                return test_type, stat, p_value, is_normal, None

        # 对自变量和因变量进行正态性检验
        test_type_x, stat_x, p_value_x, is_normal_x, result_x = test_normality(X)
        test_type_y, stat_y, p_value_y, is_normal_y, result_y = test_normality(Y)

        # 打印自变量和因变量的正态性检验结果
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

        # 陈述线性回归的原假设
        print(f"\nNull Hypothesis: There is no significant linear relationship between '{x_var}' and '{y_var}'.")

        # 检查自变量和因变量是否符合正态分布
        if not is_normal_x or not is_normal_y:
            print(f"One or both variables are not normally distributed. Proceed with caution when interpreting the results.")
    
        # 执行线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

        # 绘制散点图和回归线
        sns.regplot(x=X, y=Y, line_kws={'color': 'red'})
        plt.title(f'Regression: {y_var} ~ {x_var}')
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.show()

        # 输出回归分析结果并进行解释
        print("\nRegression Analysis Results:")
        print(f"Slope: {slope:.4f}")
        print("Interpretation: The slope indicates the strength and direction of the relationship between the variables.")

        print(f"Intercept: {intercept:.4f}")
        print(f"Prediction: When the value of '{x_var}' is 0, the predicted value of '{y_var}' is {intercept:.4f}.")

        print(f"R-squared: {r_value**2:.4f}")
        print("Explanation: The R-squared value indicates the model's goodness of fit. "
              "A higher value means the model fits the data better.")

        print(f"P-value: {p_value:.15f}")
        # 根据 p-value 判断是否拒绝原假设
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
        """执行情感分析"""
        print(f"Selected column for analysis: {text_column}")
        print(f"Data type: {self.df[text_column].dtype}")  # 列出数据类型

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
        """VADER 分析"""
        results = []
        for text in self.df[column].fillna(""):
            score = self.analyzer.polarity_scores(text)['compound']
            sentiment = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
            results.append({"Text": text, "Score": score, "Sentiment": sentiment})
        result_df = pd.DataFrame(results)
        print("\nVADER Sentiment Analysis Results:")
        print(result_df)
        return result_df  # 返回结果以便后续处理

    def textblob_analysis(self, column):
        """TextBlob 分析"""
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
        return result_df  # 返回结果以便后续处理

    def distilbert_analysis(self, column):
        """DistilBERT 分析"""
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
        return result_df  # 返回结果以便后续处理

