library(dplyr)
library(MASS)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(car)
library(ROCR)
library(dplyr)
library(tree)
library(randomForest)
library(e1071)
library(caret)

ad = read.csv('C:/Users/gagss/OneDrive/Documents/spring 2024/Data Analytics Applications/data/advertising.csv',sep=',')

head(ad)
str(ad)

summary(ad)

ba=na.omit(ad)
str(ad)

################# Data Cleaning #######################################################################

#1.
ad$Clicked.on.Ad <- as.factor(ad$Clicked.on.Ad)
ad$Male <- as.factor(ad$Male)
str(ad)

numeric_ad <- ad[, sapply(ad, is.numeric)]
# Compute the correlation matrix
M <- cor(numeric_ad)
# View the correlation matrix
print(M)
corrplot(M, method = c("number"))

ad_n <- ad[, !(names(ad) %in% c("Ad.Topic.Line", "City", "Country"))]

str(ad_n)

ad_n$Timestamp <- as.POSIXct(ad_n$Timestamp, format="%m/%d/%Y %H:%M")

# Extract day of the week, month, date, and hour of the day
ad_n$Day_of_Week <- weekdays(ad_n$Timestamp)
ad_n$Month <- format(ad_n$Timestamp, "%B")
ad_n$Date <- as.numeric(format(ad_n$Timestamp, "%d"))
ad_n$Hour_of_Day <- as.numeric(format(ad_n$Timestamp, "%H"))

numeric_ad1 <- ad_n[, sapply(ad_n, is.numeric)]
# Compute the correlation matrix
N <- cor(numeric_ad1)
# View the correlation matrix
print(N)
corrplot(N, method = c("number"))

table(ad_n$Month)
table(ad_n$Day_of_Week)

levels(ad_n$Day_of_Week)
ad_n$Day_of_Week <- factor(ad_n$Day_of_Week, levels = c("Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday"))
levels(ad_n$Month)
ad_n$Month <- factor(ad_n$Month, levels = c("January", "February", "March", "April", "May", "June", "July"))

ad_n <- ad_n[, !(names(ad_n) %in% c("Timestamp"))]

ad_s<- sample(nrow(ad_n),0.8*nrow(ad_n),replace = F) # Setting training sample to be 80% of the data
adtrain <- ad_n[ad_s,]
adtest <- ad_n[-ad_s,]

str(ad_n)

numeric_ad1 <- ad_n[, sapply(ad_n, is.numeric)]
# Compute the correlation matrix
N <- cor(numeric_ad1)
# View the correlation matrix
corrplot(N, method = c("number"))


par(mfrow=c(3,3))
boxplot(Daily.Time.Spent.on.Site ~ Clicked.on.Ad, data=ad_n, ylab='Daily.Time.Spent.on.Site', xlab='Clicked.on.Ad', col='#FF0000')
boxplot(Age ~ Clicked.on.Ad, data=ad_n, ylab='Age', xlab='Clicked.on.Ad', col='#FF3300')
boxplot(Area.Income ~ Clicked.on.Ad, data=ad_n, ylab='Area.Income', xlab='Clicked.on.Ad', col='#CC9933')
boxplot(Daily.Internet.Usage ~ Clicked.on.Ad, data=ad_n, ylab='Daily.Internet.Usage', xlab='Clicked.on.Ad', col='#33CC00')
boxplot(Male ~ Clicked.on.Ad, data=ad_n, ylab='Male', xlab='Clicked.on.Ad', col='#99CCFF')
boxplot(Day_of_Week ~ Clicked.on.Ad, data=ad_n, ylab='Day_of_Week', xlab='Clicked.on.Ad', col='#0000CC')
boxplot(Month ~ Clicked.on.Ad, data=ad_n, ylab='Month', xlab='Clicked.on.Ad', col='#9933CC')
boxplot(Date ~ Clicked.on.Ad, data=ad_n, ylab='Date', xlab='Clicked.on.Ad', col='#9900FF')
boxplot(Hour_of_Day ~ Clicked.on.Ad, data=ad_n, ylab='Hour_of_Day', xlab='Clicked.on.Ad', col='#6600FF')


#######################################################################################################

############## Logistic regression ##############

m1.log = glm(Clicked.on.Ad ~ ., data = adtrain, family = binomial)
summary(m1.log)
vif(m1.log)

predprob_log <- predict.glm(m1.log, adtest, type = "response")  
predclass_log = ifelse(predprob_log >= 0.5,1,0)

caret::confusionMatrix(as.factor(predclass_log), adtest$Clicked.on.Ad, positive = "1")


############# Decision Tree ################

# Build decision tree
m1.tree <- tree(Clicked.on.Ad ~ ., data = adtrain)

# Make predictions
predclass_tree <- predict(m1.tree, adtest, type = "class")

# Compute confusion matrix
conf_matrix_tree <- confusionMatrix(predclass_tree, adtest$Clicked.on.Ad)

# View confusion matrix
conf_matrix_tree

# Train a decision tree model using rpart
tree_model <- rpart(Clicked.on.Ad ~ ., data = adtrain)

# Check if variable importance is available in the model
if (!is.null(tree_model$variable.importance)) {
  # Convert variable importance to data frame
  feature_importance <- data.frame(
    Variable = names(tree_model$variable.importance),
    Importance = as.numeric(tree_model$variable.importance)
  )
  
  # Order the importance data frame in descending order
  feature_importance <- feature_importance[order(-feature_importance$Importance), ]
  
  # Plot the feature importance using ggplot2
  ggplot(feature_importance, aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "#C89EE3") +
    coord_flip() +  # Rotate the chart for better visibility
    labs(
      title = "Feature Importance for Decision Tree Model",
      x = "Variables",
      y = "Importance"
    ) +
    theme_minimal()
} else {
  # Print a message if variable importance is not available
  print("Variable importance is not available in the decision tree model.")
}

################ SVM ####################

# Build SVM model
m1.svm <- svm(Clicked.on.Ad ~ ., data = adtrain, kernel = "radial")

# Make predictions
predclass_svm <- predict(m1.svm, adtest)

# Compute confusion matrix
conf_matrix_svm <- confusionMatrix(predclass_svm, adtest$Clicked.on.Ad)

# View confusion matrix
conf_matrix_svm

############## Random Forest #############

# Build Random Forest model
m1.rf <- randomForest(Clicked.on.Ad ~ ., data = adtrain)

# Make predictions
predclass_rf <- predict(m1.rf, adtest)

# Compute confusion matrix
conf_matrix_rf <- confusionMatrix(predclass_rf, adtest$Clicked.on.Ad)

# View confusion matrix
conf_matrix_rf

# Build Random Forest model
m1.rf <- randomForest(Clicked.on.Ad ~ ., data = adtrain)

# Calculate variable importance
importance_rf <- importance(m1.rf)

# View variable importance
print(importance_rf)

# Plot the variable importance
# Convert importance data to a data frame
importance_df <- data.frame(
  Variable = rownames(importance_rf),
  Importance = importance_rf[, "MeanDecreaseGini"]
)

# Order the data frame by importance
importance_df <- importance_df[order(-importance_df$Importance), ]

# Plot with ggplot2
library(ggplot2)

ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#93DFCE") +
  coord_flip() +  # Rotate the plot for horizontal orientation
  labs(title = "Feature Importance in Random Forest Model",
       x = "Features",
       y = "Mean Decrease Gini") +
  theme_minimal() +  # Use a clean and minimalistic theme
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10)
  )

#######################################################################################################

#2.	I would like to see if there is any correlation with time spent on the site and 'clicked on Ad'.
# Calculate correlation

# Load necessary packages
library(dplyr)
library(ggplot2)
library(psych)

# Convert 'Clicked.on.Ad' from factor to numeric
ad$Clicked.on.Ad <- as.numeric(as.character(ad$Clicked.on.Ad))

# Calculate Pearson correlation coefficient
correlation_test <- cor.test(ad$Daily.Time.Spent.on.Site, ad$Clicked.on.Ad, method = "pearson")

# Print Pearson correlation coefficient and p-value
print(correlation_test)

library(car)  # For Levene's test

# Check normality of daily time spent on site for each group
shapiro_test_0 <- shapiro.test(ad$Daily.Time.Spent.on.Site[ad$Clicked.on.Ad == 0])
shapiro_test_1 <- shapiro.test(ad$Daily.Time.Spent.on.Site[ad$Clicked.on.Ad == 1])

# Print Shapiro-Wilk test results
print(shapiro_test_0)
print(shapiro_test_1)

# Check homoscedasticity (equal variances)
levene_test <- leveneTest(Daily.Time.Spent.on.Site ~ as.factor(Clicked.on.Ad), data = ad)
print(levene_test)

# Perform the Mann-Whitney U test
wilcox_test <- wilcox.test(ad$Daily.Time.Spent.on.Site[ad$Clicked.on.Ad == 0],
                           ad$Daily.Time.Spent.on.Site[ad$Clicked.on.Ad == 1])

# Print the result
print(wilcox_test)

# Boxplot of Daily Time Spent on Site by Clicked on Ad
ggplot(ad, aes(x = as.factor(Clicked.on.Ad), y = Daily.Time.Spent.on.Site, fill = as.factor(Clicked.on.Ad))) +
  geom_boxplot() +
  labs(x = "Clicked on Ad", y = "Daily Time Spent on Site", fill = "Clicked on Ad") +
  ggtitle("Boxplot of Daily Time Spent on Site by Clicked on Ad") +
  theme_minimal() +
  scale_fill_manual(values = c("purple", "magenta"))  # Customize colors for "0" and "1"


#######################################################################################################

#3.	What is the average time spent on the site daily and average area income of users who click on the ad.

clicked_on_ad_data <- ad[ad$Clicked.on.Ad == "1", ]

# Calculate the average time spent on the site daily
average_time_spent <- mean(clicked_on_ad_data$Daily.Time.Spent.on.Site)

# Calculate the average area income
average_area_income <- mean(clicked_on_ad_data$Area.Income)

# Print the results
cat("Average time spent on the site daily by users who clicked on the ad:", average_time_spent, "\n")
cat("Average area income of users who clicked on the ad:", average_area_income, "\n")

# Calculate and print the minimum and maximum values of "Daily.Time.Spent.on.Site"
min_time_spent <- min(clicked_on_ad_data$Daily.Time.Spent.on.Site)
max_time_spent <- max(clicked_on_ad_data$Daily.Time.Spent.on.Site)
cat("Minimum time spent on the site daily:", min_time_spent, "\n")
cat("Maximum time spent on the site daily:", max_time_spent, "\n")

# Calculate and print the minimum and maximum values of "Area.Income"
min_area_income <- min(clicked_on_ad_data$Area.Income)
max_area_income <- max(clicked_on_ad_data$Area.Income)
cat("Minimum area income:", min_area_income, "\n")
cat("Maximum area income:", max_area_income, "\n")


#######################################################################################################

#4.	I would like to investigate if mean income changes with click on the ad. 
#To investigate if the mean income changes with clicking on the ad, you can perform a hypothesis test, such as a two-sample t-test or a Mann-Whitney U test, depending on the distribution of the data and the assumptions you're willing to make.
#The null hypothesis for this test is that there is no difference in mean income between users who clicked on the ad and those who did not. If the p-value is less than your chosen significance level (e.g., 0.05), you would reject the null hypothesis, suggesting that there is a significant difference in mean income between the two groups. Otherwise, you would fail to reject the null hypothesis, indicating that there is not enough evidence to conclude that the mean income differs between the two groups.


# Check normality of daily time spent on site for each group
shapiro_test_0 <- shapiro.test(ad$Area.Income[ad$Clicked.on.Ad == 0])
shapiro_test_1 <- shapiro.test(ad$Area.Income[ad$Clicked.on.Ad == 1])

# Print Shapiro-Wilk test results
print(shapiro_test_0)
print(shapiro_test_1)

# Check homoscedasticity (equal variances)
levene_test <- leveneTest(Area.Income ~ as.factor(Clicked.on.Ad), data = ad)
print(levene_test)

# Perform the Wilcoxon rank sum test (Mann-Whitney U test)
wilcox_test <- wilcox.test(
  ad$Area.Income[ad$Clicked.on.Ad == 0],
  ad$Area.Income[ad$Clicked.on.Ad == 1],
  alternative = "two.sided"
)

# Print the test result
print(wilcox_test)


# Separate the data into two groups: clicked on ad and not clicked on ad
clicked_on_ad_data <- ad[ad$Clicked.on.Ad == "1", ]
not_clicked_on_ad_data <- ad[ad$Clicked.on.Ad == "0", ]

# Perform a two-sample t-test to compare the mean income between the two groups
t_test_result <- t.test(clicked_on_ad_data$Area.Income, not_clicked_on_ad_data$Area.Income)

# Print the result
print(t_test_result)

# p-value < 2.2e-16

data <- rbind(
  cbind(clicked_on_ad_data, group = "Clicked on Ad"),
  cbind(not_clicked_on_ad_data, group = "Did Not Click on Ad")
)

# Create a boxplot comparing the income of the two groups
ggplot(data, aes(x = group, y = Area.Income, fill = group)) +
  geom_boxplot() +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black", position = position_dodge(width = 0.75)) +
  labs(title = "Comparison of Area Income Between Clicked on Ad and Did Not Click on Ad",
       x = "Group",
       y = "Area Income") +
  theme_minimal() +
  scale_fill_manual(values = c("Clicked on Ad" = "#B027BB", "Did Not Click on Ad" = "red")) +
  theme(legend.position = "none")  # Remove the legend


#######################################################################################################

# 5.	I will check if total daily internet use and time spent by users on the site are related to each other in some way.
#To check if total daily internet use and time spent by users on the site are related to each other, you can perform a correlation analysis. Pearson correlation coefficient can be used to measure the strength and direction of the linear relationship between two continuous variables.

correlation <- cor(ad$Daily.Internet.Usage, ad$Daily.Time.Spent.on.Site)

print(correlation)

# [1] 0.5186585

# 1 indicates a perfect positive linear relationship,

ggplot(ad, aes(x = Daily.Internet.Usage, y = Daily.Time.Spent.on.Site)) +
  geom_point(color = "green", alpha = 0.6) +  # Use blue color for points and adjust transparency
  geom_smooth(method = "lm", se = TRUE, color = "red") +  # Add a linear regression line in red with confidence interval
  labs(title = "Scatter Plot of Daily Internet Usage vs. Daily Time Spent on Site",
       x = "Daily Internet Usage (minutes per day)",
       y = "Daily Time Spent on Site (minutes per day)") +
  theme_minimal() +  # Use a minimal theme for a clean look
  theme(plot.title = element_text(hjust = 0.5),  # Center the plot title
        plot.background = element_rect(fill = "white"),  # Set white background
        panel.grid = element_blank())  # Remove grid lines

#######################################################################################################

#6.
library(ggplot2)

# Subset data for users who clicked on the ad
clicked_users <- subset(ad, Clicked.on.Ad == "1")

# Summary statistics for numerical variables
summary_stats <- summary(clicked_users[, c("Age", "Area.Income", "Daily.Time.Spent.on.Site", "Daily.Internet.Usage")])
print(summary_stats)

# Frequency table for categorical variables
gender_freq <- table(clicked_users$Male)
gender_freq

# Data Visualization
# Histograms for numerical variables
ggplot(clicked_users, aes(x = Age)) + geom_histogram(binwidth = 5, fill = "#E32D91", color = "black") + labs(title = "Age Distribution")
ggplot(clicked_users, aes(x = Area.Income)) + geom_histogram(binwidth = 5000, fill = "#C830CC", color = "black") + labs(title = "Area Income")
ggplot(clicked_users, aes(x = Daily.Time.Spent.on.Site)) + geom_histogram(binwidth = 5, fill = "#4EA6DC", color = "black") + labs(title = "Daily Time Spent on Site")
ggplot(clicked_users, aes(x = Daily.Internet.Usage)) + geom_histogram(binwidth = 20, fill = "#8971E1", color = "black") + labs(title = "Daily Internet Usage Distribution")

# Bar plots for categorical variables
ggplot(clicked_users, aes(x = factor(Male))) + geom_bar(fill = "#4775E7") + labs(title = "Gender")

#######################################################################################################

#7.	Which day, which month, which date, what time of the day do users usually click on the ad. 

ad$Timestamp <- as.POSIXct(ad$Timestamp, format="%m/%d/%Y %H:%M")

# Extract day of the week, month, date, and hour of the day
ad$Day_of_Week <- weekdays(ad$Timestamp)
ad$Month <- format(ad$Timestamp, "%B")
ad$Date <- as.numeric(format(ad$Timestamp, "%d"))
ad$Hour_of_Day <- as.numeric(format(ad$Timestamp, "%H"))

# Group by day of the week, month, date, and hour of the day, and count ad clicks
clicks_by_day_of_week <- ad %>% group_by(Day_of_Week) %>% summarise(Clicks = n())
clicks_by_month <- ad %>% group_by(Month) %>% summarise(Clicks = n())
clicks_by_date <- ad %>% group_by(Date) %>% summarise(Clicks = n())
clicks_by_hour_of_day <- ad %>% group_by(Hour_of_Day) %>% summarise(Clicks = n())

ad$Day_of_Week <- factor(ad$Day_of_Week, levels = c("Sunday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday","Saturday"))
ad$Month <- factor(ad$Month, levels = c("January", "February", "March", "April", "May", "June", "July"))

# Plotting
par(mfrow=c(2,2))

# Plot clicks by day of the week
barplot(clicks_by_day_of_week$Clicks, names.arg=clicks_by_day_of_week$Day_of_Week, col="#2FA3EE", main="Ad Clicks by Day of the Week", xlab="Day of the Week", ylab="Number of Clicks")

# Plot clicks by month
barplot(clicks_by_month$Clicks, names.arg=clicks_by_month$Month, col="#4BCAAD", main="Ad Clicks by Month", xlab="Month", ylab="Number of Clicks")

# Plot clicks by date
plot(clicks_by_date$Date, clicks_by_date$Clicks, type="o", col="black", main="Ad Clicks by Date", xlab="Date", ylab="Number of Clicks")

# Fill the area under the line graph with a light blue color
polygon(c(clicks_by_date$Date, rev(clicks_by_date$Date)), 
        c(rep(0, length(clicks_by_date$Clicks)), rev(clicks_by_date$Clicks)), 
        col="#86C157", border=NA)

# Plot the line graph again to make sure it is on top of the fill
lines(clicks_by_date$Date, clicks_by_date$Clicks, col="black")
# Plot clicks by hour of the day
barplot(clicks_by_hour_of_day$Clicks, names.arg=clicks_by_hour_of_day$Hour_of_Day, col="#D99C3F", main="Ad Clicks by Hour of the Day", xlab="Hour of the Day", ylab="Number of Clicks")

########################################################################################################

#8.	If we divide the time into morning, afternoon and night, which part of the day do users click on ad?  

ad$Clicked.on.Ad <- as.factor(ad$Clicked.on.Ad)

# Extract hour of the day
ad$Hour_of_Day <- as.numeric(format(ad$Timestamp, "%H"))

# Categorize hours into morning, afternoon, and night
ad$Time_of_Day <- cut(ad$Hour_of_Day, breaks=c(0, 12, 18, 24), labels=c("Morning", "Afternoon", "Night"), right=FALSE)

# Count ad clicks by time of the day
clicks_by_time_of_day <- table(ad$Time_of_Day)

# Print the count of ad clicks by time of the day
print(clicks_by_time_of_day)

contingency_table <- table(ad$Clicked.on.Ad, ad$Time_of_Day)

# Perform chi-square test
chi_sq_result <- chisq.test(contingency_table)

# Print the result
print(chi_sq_result)

#This code will perform a chi-square test of independence to determine whether the distribution of ad clicks is significantly different across different parts of the day. The null hypothesis is that there is no association between the time of day and ad clicks. If the p-value is less than a chosen significance level (e.g., 0.05), you reject the null hypothesis, indicating that there is a statistically significant association
# p-value = 0.3157
#With a p-value of 0.3157, it suggests that there is no significant association between the time of day and ad clicks at the chosen significance level (usually 0.05). Therefore, based on this analysis, we fail to reject the null hypothesis, indicating that the distribution of ad clicks across different parts of the day is not significantly different.


# Convert the table to a data frame
clicks_by_time_of_day_df <- as.data.frame(clicks_by_time_of_day)

# Plot the count of ad clicks by time of the day
ggplot(clicks_by_time_of_day_df, aes(x = Var1, y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", fill = c("#4BCAAD", "orange", "coral")) +
  labs(title = "Ad Clicks by Time of Day", x = "Time of Day", y = "Count of Clicks") +
  theme_minimal()





