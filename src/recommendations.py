def generate_retention_recommendations(customer_dict: dict, churn_proba: float) -> list:
    recommendations = []
    
    if churn_proba < 0.30:
        recommendations.append("Customer exhibits low churn risk. Maintain standard service quality and engagement.")
        return recommendations
        
    # Month-to-month rule
    if customer_dict.get('Contract') == 'Month-to-month':
        recommendations.append("📌 **Contract Retention:** Customer is on a Month-to-Month plan. Offer a 15% discount or service upgrade on switching to a 1-year or 2-year contract.")
        
    # High charges rule
    if customer_dict.get('MonthlyCharges', 0) > 70.0:
        recommendations.append("📌 **Pricing Optimization:** Customer has high monthly charges. Suggest a bundled plan or loyalty discount to reduce churn probability.")
        
    # Fiber Optic rule
    if customer_dict.get('InternetService') == 'Fiber optic':
        recommendations.append("📌 **Technical Quality Check:** Fiber optic users show higher attrition rates. Recommend a proactive connection stability check or free premium tech support.")
        
    # Payment Method rule
    if customer_dict.get('PaymentMethod') == 'Electronic check':
        recommendations.append("📌 **Automated Payment Incentive:** Encourage switching from Electronic Check to Auto-pay (Bank/Credit Card) with a one-time bill credit.")
        
    return recommendations