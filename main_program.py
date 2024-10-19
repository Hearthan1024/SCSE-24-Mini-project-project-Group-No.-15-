from analysis_module import DataInspection, DataAnalysis, SentimentAnalysis


def main():
    di = DataInspection()

    path = input("Enter the path to the dataset: ")
    di.load_csv(path)

    if di.df is None:
        print("Failed to load the dataset. Exiting...")
        return

    df = di.df
    analysis = DataAnalysis(df)
    sentiment = SentimentAnalysis(df)

    while True:
        print("\nHow do you want to analyze your data?")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")

        choice = input("Enter your choice (1 – 7): ")

        if choice == '1':
            while True:
                di.show_plot_variables()
                var_choice = input("Enter your choice:")

                try:
                    var_index = int(var_choice) - 1

                    if 0 <= var_index < len(di.df.columns):
                        variable = di.df.columns[var_index]
                        di.plot_distribution(variable) 
                    elif var_choice == str(len(di.df.columns) + 1):
                        break  
                    elif var_choice == str(len(di.df.columns) + 2):
                        exit() 
                    else:
                        print("Invalid choice. Please try again.")
                except (ValueError, IndexError):
                    print("Invalid choice. Please enter a valid number.")
                    
        elif choice == '2':
            analysis.show_anova_variables()
            continuous_var = input("Enter a continuous (interval/ratio) variable: ")
            categorical_var = input("Enter a categorical (ordinal/nominal) variable: ")
            analysis.anova_analysis(continuous_var, categorical_var)
            
        elif choice == '3':
            print("\nFollowing variables are Interval:")
            interval_vars = analysis.show_variables('Ratio')
            continuous_var = input("Enter an Interval variable: ")

            print("\nFollowing variables are Nominal:")
            nominal_vars = analysis.show_variables('Nominal')
            categorical_var = input("Enter a Nominal variable: ")

            analysis.t_test_or_mannwhitney(continuous_var, categorical_var)
            
        elif choice == '4':
            print("\nAvailable categorical variables for Chi-Square Test:")
            categorical_vars = analysis.show_categorical_variables()
            var1 = input("Enter the first variable: ")
            var2 = input("Enter the second variable: ")

            if var1 not in categorical_vars or var2 not in categorical_vars:
                print("Invalid variable selection. Please try again.")
                continue

            analysis.chi_square_test(var1, var2)
            
        elif choice == '5':
            print("\nAvailable continuous (Interval) variables:")
            interval_vars = analysis.show_interval_variables()

            x_var = input("Enter the X (independent) variable: ")
            y_var = input("Enter the Y (dependent) variable: ")

            if x_var not in interval_vars or y_var not in interval_vars:
                print("Invalid variable selection. Please try again.")
                continue

            analysis.perform_regression(x_var, y_var)
            
        elif choice == '6':
            print("\nLooking for text data in your dataset...")

            text_columns = df.select_dtypes(include=['object']).columns
    
            if len(text_columns) == 0:
                print("Sorry, your dataset does not have a suitable length text data.")
                print("Therefore, Sentiment Analysis is not possible.")
                print("Returning to previous menu...")
                continue

            print("\nAvailable columns for sentiment analysis (text-based columns):")
            for col in text_columns:
                print(f"- {col}")

            column_name = input("Enter the column name to be analyzed: ")
            if column_name not in text_columns:
                print("Invalid column name! Please try again.")
                continue

            print("Choose the type of sentiment analysis:")
            print("1. VADER (for short texts)")
            print("2. TextBlob (provides Polarity and Subjectivity)")
            print("3. DistilBERT (requires transformers library)")
            analysis_method = input("Enter your choice (1, 2, or 3): ")

            result_df = sentiment.analyze(column_name, analysis_method)

            if result_df is not None:
                save_choice = input("Do you want to save the results to a CSV file? (y/n): ")
                if save_choice.lower() == 'y':
                    result_df.to_csv(f"sentiment_analysis_{column_name}.csv", index=False)
                    print(f"Results saved to 'sentiment_analysis_{column_name}.csv'.")

        
        elif choice == '7':
            break

if __name__ == "__main__":
    main()
