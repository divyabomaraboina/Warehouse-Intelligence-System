import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import load_gold

def load_data():
    # load gold data
    df= load_gold()
    df['MONTH'] = df['DATE'].dt.month
    print(f"Total rows: {len(df):,}")
    print(f"Date range : {df['DATE'].min()} to {df['DATE'].max()}")
    return df

def define_groups(df):
    # Group A → non-summer months (9,10,11,12,1,2,3,4)
    Summer_months = [5, 6, 7, 8]
    # Group B → summer months (5,6,7,8)
    group_b = df[df['MONTH'].isin(Summer_months)]
    group_a= df[~df['MONTH'].isin(Summer_months)]
    print(f"Group A (non-summer months) rows: {len(group_a):,}")
    print(f"Group B (summer months) rows: {len(group_b):,}")
    print(f"Group A avg sales: {group_a['WAREHOUSE SALES'].mean():,.2f}")
    print(f"Group B avg sales: {group_b['WAREHOUSE SALES'].mean():,.2f}")
    return group_a, group_b


def run_ttest(group_a, group_b):
    sales_a = group_a['WAREHOUSE SALES']
    sales_b = group_b['WAREHOUSE SALES']
    t_stat,p_value = stats.ttest_ind(sales_a,sales_b)
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.2e}")
    if p_value < 0.05:
        print("Reject null hypothesis: Significant difference in sales between groups.")
        print("Summer effect is Real , not just by chance.")
    else:
        print("Fail to reject null hypothesis: No significant difference in sales between groups.")
        print("Summer effect is not statistically significant.")
    return t_stat, p_value
def controlled_analysis(df):
    top_suppliers = df.groupby('SUPPLIER')['WAREHOUSE SALES']\
    .sum().sort_values(ascending=False).head(3).index.tolist()
    print(f"Removing top suppliers: {top_suppliers}")
    df_controlled = df[~df['SUPPLIER'].isin(top_suppliers)]
    summer = df_controlled[df_controlled['MONTH'].isin([5,6,7,8])]
    non_summer = df_controlled[~df_controlled['MONTH'].isin([5,6,7,8])]
    # T-test run
    controlled_t_stat, controlled_p_value = stats.ttest_ind(summer['WAREHOUSE SALES'], non_summer['WAREHOUSE SALES'])
    print(f"\n=== CONTROLLED ANALYSIS ===")
    print(f"Rows after removing top suppliers: {len(df_controlled):,}")
    print(f"T-statistic: {controlled_t_stat:.4f}")
    print(f"P-value: {controlled_p_value:.2e}")

    if controlled_p_value < 0.05:
        print("Reject null hypothesis: Significant difference in sales without top suppliers.")
        print("Summer effect is Real , not just by chance.")
    else :
        print("Fail to reject null hypothesis: No significant difference in sales without top suppliers.")
        print("Summer effect is not statistically significant.")
    return controlled_t_stat, controlled_p_value

def segmented_analysis(df):
    # Segment by item type
    summer_months = [5, 6, 7, 8]
    results = []
    item_types = df['ITEM TYPE'].unique()
    print("\n=== SEGMENTED ANALYSIS BY ITEM TYPE ===")
    print(f"{'Item Type':<15} {'Non-Summer':>12} {'Summer':>12} {'p-value':>12} {'Significant':>12}")
    print("-" * 65)

    for item in item_types:
        item_df = df[df['ITEM TYPE'] == item]
        summer = item_df[item_df['MONTH'].isin(summer_months)]['WAREHOUSE SALES']
        non_summer= item_df[~item_df['MONTH'].isin(summer_months)]['WAREHOUSE SALES']
        
        if len(summer) <30 or len(non_summer) <30:
            continue
        t_stat, p_value = stats.ttest_ind(summer,
                                           non_summer)
        significant = "Yes" if p_value < 0.05 else "No"
        print(f"{item:<15} "
              f"${non_summer.mean():>11,.2f} "
              f"${summer.mean():>11,.2f} "
              f"{p_value:>12.2e} "
              f"{significant:>12}")
        results.append({
            'item_type': item,
            'non_summer_avg': non_summer.mean(),
            'summer_avg': summer.mean(),
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    results_df = pd.DataFrame(results)
    return results_df
def monthly_aggregate_test(df):
    # group by DATE → monthly totals
    monthly = df.groupby('DATE')['WAREHOUSE SALES'].sum().reset_index()
    
    # month number extract cheyyi
    monthly['MONTH'] = monthly['DATE'].dt.month
      # Add this line inside function
    print(f"Unique dates: {sorted(monthly['DATE'].tolist())}")
    print(f"Total unique months: {len(monthly)}")
    
    # summer groups — Series directly
    summer_months = [5, 6, 7, 8]
    summer = monthly[monthly['MONTH'].isin(summer_months)]['WAREHOUSE SALES']
    non_summer = monthly[~monthly['MONTH'].isin(summer_months)]['WAREHOUSE SALES']
    
    print(f"\n=== MONTHLY AGGREGATE TEST ===")
    print(f"Total months:      {len(monthly)}")
    print(f"Summer months:     {len(summer)}")
    print(f"Non-summer months: {len(non_summer)}")
    
    # .mean() directly — no ['WAREHOUSE SALES'] again
    print(f"Summer avg:        ${summer.mean():,.0f}")
    print(f"Non-summer avg:    ${non_summer.mean():,.0f}")
    
    # pass Series directly — no ['WAREHOUSE SALES'] again
    t_stat_monthly, p_value_monthly = stats.ttest_ind(summer, non_summer)
    
    print(f"T-statistic: {t_stat_monthly:.4f}")
    print(f"P-value:     {p_value_monthly:.2e}")
    
    if p_value_monthly < 0.05:
        print("✓ Monthly aggregate — summer effect REAL!")
    else:
        print("✗ Effect disappears at monthly level!")
        print("Previous result was large sample size artifact!")
    
    return t_stat_monthly, p_value_monthly
def calculate_effect_size(group_a, group_b, df):
    
    # Transaction level
    mean_a = group_a['WAREHOUSE SALES'].mean()
    mean_b = group_b['WAREHOUSE SALES'].mean()
    std_a = group_a['WAREHOUSE SALES'].std()
    std_b = group_b['WAREHOUSE SALES'].std()
    n_a = len(group_a)
    n_b = len(group_b)
    
    pooled_std = np.sqrt(
        ((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a+n_b-2)
    )
    d_transaction = (mean_b - mean_a) / pooled_std
    
    # Monthly level — more meaningful
    monthly = df.groupby('DATE')['WAREHOUSE SALES'].sum().reset_index()
    monthly['MONTH'] = monthly['DATE'].dt.month
    
    summer_m = monthly[monthly['MONTH'].isin([5,6,7,8])]['WAREHOUSE SALES']
    non_summer_m = monthly[~monthly['MONTH'].isin([5,6,7,8])]['WAREHOUSE SALES']
    
    pooled_std_m = np.sqrt(
        ((len(summer_m)-1)*summer_m.std()**2 + 
         (len(non_summer_m)-1)*non_summer_m.std()**2) / 
        (len(summer_m)+len(non_summer_m)-2)
    )
    d_monthly = (summer_m.mean() - non_summer_m.mean()) / pooled_std_m
    
    print(f"\n=== EFFECT SIZE ===")
    print(f"Transaction level Cohen's d: {d_transaction:.4f} → Small")
    print(f"Monthly level Cohen's d:     {d_monthly:.4f}")
    
    if abs(d_monthly) < 0.2:
        interp = "Small"
    elif abs(d_monthly) < 0.5:
        interp = "Medium"
    elif abs(d_monthly) < 0.8:
        interp = "Large"
    else:
        interp = "Very Large"
    
    print(f"Monthly interpretation:      {interp}")
    print(f"\nBusiness impact:")
    print(f"Extra revenue per summer month: ${summer_m.mean()-non_summer_m.mean():,.0f}")
    print(f"Total summer premium (4 months): ${(summer_m.mean()-non_summer_m.mean())*4:,.0f}")
    
    return d_transaction, d_monthly
def visualize_results(group_a, group_b, results_df):
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1 → Box plot — summer vs non-summer
    axes[0].boxplot(
        [group_a['WAREHOUSE SALES'], group_b['WAREHOUSE SALES']],
        labels=['Non-Summer', 'Summer']
    )
    axes[0].set_title('Warehouse Sales Distribution')
    axes[0].set_ylabel('Sales ($)')
    axes[0].set_xlabel('Season')
    
    # Plot 2 → Bar chart — item type comparison
    x = range(len(results_df))
    width = 0.35
    axes[1].bar(x, results_df['non_summer_avg'], 
                width, label='Non-Summer', color='steelblue')
    axes[1].bar([i+width for i in x], results_df['summer_avg'],
                width, label='Summer', color='orange')
    axes[1].set_xticks([i+width/2 for i in x])
    axes[1].set_xticklabels(results_df['item_type'], rotation=45)
    axes[1].set_title('Summer vs Non-Summer by Item Type')
    axes[1].set_ylabel('Average Sales ($)')
    axes[1].legend()
    
    # Plot 3 → p-values by item type
    colors = ['green' if p < 0.05 else 'red' 
              for p in results_df['p_value']]
    axes[2].bar(results_df['item_type'], 
                results_df['p_value'], 
                color=colors)
    axes[2].axhline(y=0.05, color='black', 
                    linestyle='--', label='p=0.05 threshold')
    axes[2].set_title('P-values by Item Type')
    axes[2].set_ylabel('P-value')
    axes[2].set_xlabel('Item Type')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('models/ab_testing_results.png')
    plt.show()
    print("Visualization saved!")
def generate_report(group_a, group_b, p_value, d_monthly, results_df, controlled_p, df):
    
    # calculate everything dynamically
    summer_avg     = group_b['WAREHOUSE SALES'].mean()
    non_summer_avg = group_a['WAREHOUSE SALES'].mean()
    diff_pct       = ((summer_avg - non_summer_avg) / non_summer_avg) * 100
    
    # monthly impact — dynamically calculate cheyyi
    monthly = df.groupby('DATE')['WAREHOUSE SALES'].sum().reset_index()
    monthly['MONTH'] = monthly['DATE'].dt.month
    
    summer_monthly     = monthly[monthly['MONTH'].isin([5,6,7,8])]['WAREHOUSE SALES']
    non_summer_monthly = monthly[~monthly['MONTH'].isin([5,6,7,8])]['WAREHOUSE SALES']
    
    monthly_summer_avg     = summer_monthly.mean()
    monthly_non_summer_avg = non_summer_monthly.mean()
    monthly_diff           = monthly_summer_avg - monthly_non_summer_avg
    annual_impact          = monthly_diff * 4
    
    # seasonal items
    seasonal_items     = results_df[results_df['significant']==True]['item_type'].tolist()
    non_seasonal_items = results_df[results_df['significant']==False]['item_type'].tolist()
    
    
    # Check if summer effect still significant
    if p_value > 0.05:
        print("⚠️ WARNING: Summer effect no longer significant!")
        print("→ Do NOT increase summer stock")
        print("→ Pattern has changed — investigate why")
        return
    
    # Check effect size
    if d_monthly < 0.5:
        print("⚠️ WARNING: Summer effect weakening!")
        print(f"→ Cohen's d dropped to {d_monthly:.2f}")
        print("→ Reduce summer stock increase by 50%")
        return
    
    # Effect still strong — normal recommendations
    seasonal_items = results_df[
        results_df['significant']==True
    ]['item_type'].tolist()
    
    if not seasonal_items:
        print("⚠️ No seasonal items found!")
        print("→ Year-round flat ordering strategy")
        return
    
    print("✅ Summer effect confirmed — recommendations:")
    print(f"→ Stock {', '.join(seasonal_items)} by April")
    print(f"→ Effect size: {d_monthly:.2f} (strong)")
    import json
        
    seasonal_items     = results_df[results_df['significant']==True]['item_type'].tolist()
    non_seasonal_items = results_df[results_df['significant']==False]['item_type'].tolist()
        
    ab_data = {
            "p_value":           float(p_value),
            "monthly_cohens_d":  float(d_monthly),
            "annual_opportunity": round(float(annual_impact), 0),
            "summer_avg":        round(float(monthly_summer_avg), 0),
            "non_summer_avg":    round(float(monthly_non_summer_avg), 0),
            "seasonal_items":    seasonal_items,
            "non_seasonal_items": non_seasonal_items
        }
        
    with open("models/ab_results.json", "w") as f:
            json.dump(ab_data, f, indent=2)
        
    print("AB results saved → models/ab_results.json")
    
if __name__ == "__main__":
    # run everything
    df= load_data()
    group_a, group_b = define_groups(df)
    t_stat, p_value = run_ttest(group_a, group_b)
    controlled_t_stat, controlled_p_value = controlled_analysis(df)
    print(df.columns.tolist()) 
    results_df = segmented_analysis(df)
    t_stat_monthly, p_value_monthly = monthly_aggregate_test(df)
    d_transaction, d_monthly = calculate_effect_size(group_a, group_b, df)
    visualize_results(group_a, group_b, results_df)
    generate_report(group_a, group_b, p_value, d_monthly, results_df, controlled_p_value, df)