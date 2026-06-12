export interface FraudPattern {
  id: string;
  label: string;
  summary: string;
  outcome: string;
  loss_usd: number;
}

export interface CartItem {
  sku: string;
  name: string;
  qty: number;
  unit_price_usd: number;
}

export interface Customer {
  email: string;
  name: string;
  account_age_days: number;
  prior_orders: number;
  link_returning: boolean;
  billing_country: string;
  shipping_country: string;
  shipping_address: string;
  ip_country: string;
}

export interface SampleOrder {
  id: string;
  label: string;
  description: string;
  cart: CartItem[];
  customer: Customer;
}

export interface IndexRecord {
  id: string;
  vector: number[];
}
