"""
Angel Intelligence - Order Context Service

Retrieves order data to provide context for call analysis validation.
Used to verify that captured data (customer details, amounts, opt-ins) matches
what was discussed during the call.

Database flow:
1. halo_config.client_tables → get orders/customers/products table names for client
2. halo_data.{orders_table} → get order details
3. halo_data.{customers_table} → get customer details (name, address, opt-ins)
4. halo_config.{products_table} → get product description
5. wings_config.client_configuration → get any field name overrides
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import mysql.connector
from mysql.connector import Error as MySQLError

from src.config import get_settings
from src.database import DatabaseConnection

logger = logging.getLogger(__name__)


# Default field mappings - maps output field name to (table_alias, field_name, expression)
# Expression uses **FIELD** as placeholder for the actual field reference
DEFAULT_FIELD_MAPPINGS = {
    # Customer personal info
    "fname": {"table": "cust", "field": "fname", "expression": ""},
    "surname": {"table": "cust", "field": "surname", "expression": ""},
    "title": {"table": "cust", "field": "title", "expression": ""},
    "address1": {"table": "cust", "field": "address1", "expression": ""},
    "address2": {"table": "cust", "field": "address2", "expression": ""},
    "address3": {"table": "cust", "field": "address3", "expression": ""},
    "town": {"table": "cust", "field": "town", "expression": ""},
    "county": {"table": "cust", "field": "county", "expression": ""},
    "postcode": {"table": "cust", "field": "postcode", "expression": ""},
    "telephone": {"table": "cust", "field": "telephone", "expression": ""},
    "email_address": {"table": "cust", "field": "email", "expression": ""},
    
    # Order details
    "orderref": {"table": "ords", "field": "orderref", "expression": ""},
    "product": {"table": "ords", "field": "product", "expression": ""},
    "amount": {"table": "ords", "field": "amount", "expression": "CAST(**FIELD** as DECIMAL(10,2))"},
    "date": {"table": "ords", "field": "date", "expression": ""},
    
    # Product description (from products table)
    "product_desc": {"table": "prod", "field": "desc", "expression": ""},
    
    # Gift aid
    "giftaid": {"table": "cust", "field": "giftaid", "expression": "CASE WHEN **FIELD** IN ('Y','E') THEN 'Yes' WHEN **FIELD** = 'N' THEN 'No' ELSE 'Unknown' END"},
    
    # Opt-ins (note: mail_out/phone_out are opt-OUT fields, so logic is inverted)
    "opt_post": {"table": "cust", "field": "mail_out", "expression": "CASE WHEN **FIELD** IN ('Y','E') THEN 'No' WHEN **FIELD** = 'N' THEN 'Yes' ELSE 'Unknown' END"},
    "opt_phone": {"table": "cust", "field": "phone_out", "expression": "CASE WHEN **FIELD** IN ('Y','E') THEN 'No' WHEN **FIELD** = 'N' THEN 'Yes' ELSE 'Unknown' END"},
    "opt_email": {"table": "cust", "field": "email_con", "expression": "CASE WHEN **FIELD** IN ('Y','E') THEN 'Yes' WHEN **FIELD** = 'N' THEN 'No' ELSE 'Unknown' END"},
    "opt_sms": {"table": "cust", "field": "sms_con", "expression": "CASE WHEN **FIELD** IN ('Y','E') THEN 'Yes' WHEN **FIELD** = 'N' THEN 'No' ELSE 'Unknown' END"},
    
    # Customer status (for filtering)
    "status": {"table": "cust", "field": "status", "expression": ""},
}


@dataclass
class OrderContext:
    """Context data for an order, used in analysis prompts."""
    orderref: str
    
    # Customer details
    customer_name: Optional[str] = None  # Formatted: "Mr John Smith"
    customer_address: Optional[str] = None  # Concatenated address
    customer_telephone: Optional[str] = None
    customer_email: Optional[str] = None
    
    # Order details
    product: Optional[str] = None
    product_description: Optional[str] = None
    amount: Optional[float] = None
    
    # Gift aid
    giftaid: Optional[str] = None  # "Yes" / "No" / "Unknown"
    
    # Opt-in preferences
    opt_post: Optional[str] = None
    opt_phone: Optional[str] = None
    opt_email: Optional[str] = None
    opt_sms: Optional[str] = None
    
    # Error tracking
    error: Optional[str] = None
    
    @property
    def has_context(self) -> bool:
        """Check if we have enough context for validation."""
        return self.customer_name is not None or self.product is not None
    
    def to_prompt_context(self) -> str:
        """
        Format order context for inclusion in analysis prompt.
        
        Returns empty string if insufficient context available.
        """
        if not self.has_context:
            return ""
        
        lines = [
            "ORDER DATA VALIDATION:",
            f"- Order Reference: {self.orderref}",
        ]
        
        # Product and amount
        if self.product_description:
            lines.append(f"- Product: {self.product_description}")
        elif self.product:
            lines.append(f"- Product Code: {self.product}")
        
        if self.amount is not None:
            lines.append(f"- Amount: £{self.amount:.2f}")
        
        if self.giftaid:
            lines.append(f"- Gift Aid: {self.giftaid}")
        
        # Customer details
        lines.append("")
        lines.append("Customer Details Captured:")
        
        if self.customer_name:
            lines.append(f"- Name: {self.customer_name}")
        
        if self.customer_address:
            lines.append(f"- Address: {self.customer_address}")
        
        if self.customer_telephone:
            lines.append(f"- Telephone: {self.customer_telephone}")
        
        if self.customer_email:
            lines.append(f"- Email: {self.customer_email}")
        
        # Opt-in preferences
        opt_ins = []
        if self.opt_post:
            opt_ins.append(f"Post: {self.opt_post}")
        if self.opt_phone:
            opt_ins.append(f"Phone: {self.opt_phone}")
        if self.opt_email:
            opt_ins.append(f"Email: {self.opt_email}")
        if self.opt_sms:
            opt_ins.append(f"SMS: {self.opt_sms}")
        
        if opt_ins:
            lines.append("")
            lines.append("Opt-in Preferences Captured:")
            lines.append(f"- {', '.join(opt_ins)}")
        
        # Validation instructions
        lines.extend([
            "",
            "DATA CAPTURE VALIDATION RULES:",
            "You MUST verify all captured data matches what the supporter actually stated. Common errors include:",
            "- Name spelling errors (agent typed what they heard, not what was spelled out)",
            "- Address captured incorrectly (wrong house number, postcode typos, missed flat/unit numbers)",
            "- Amount captured incorrectly (e.g. supporter said £10, agent entered £100 or vice versa)",
            "- Gift Aid recorded as 'Yes' when supporter declined, or 'No' when they agreed",
            "- Opt-in preferences inverted (agent ticked 'yes' when supporter said 'no' or vice versa)",
            "",
            "DATA CAPTURE VALIDATION TASK:",
            "1. Listen for when the supporter provides their personal details (name, address, phone, email)",
            "2. Compare EXACTLY what they said vs what was captured above",
            "3. Listen for the donation amount agreed and compare to the amount captured",
            "4. Listen for Gift Aid discussion - did supporter agree or decline? Does it match?",
            "5. Listen for marketing preferences - what did supporter say for post/phone/email/SMS?",
            "",
            "If ANY captured data does NOT match what was stated in the call, you MUST add a NEGATIVE score_impact with:",
            "- category: 'Data_entry_accuracy'",
            "- Clear explanation: what was captured vs what was actually stated",
            "- Include the exact quote from the supporter if possible",
        ])
        
        return "\n".join(lines)


class OrderContextService:
    """
    Service to retrieve order context for call analysis.
    
    Provides data capture validation context by:
    1. Looking up the client's table names
    2. Querying order and customer data
    3. Getting product description
    4. Handling field name variations per client
    """
    
    def __init__(self):
        """Initialise the order context service."""
        self.settings = get_settings()
        self.db = DatabaseConnection()
        
        # Database names
        self.halo_config_db = "halo_config"
        self.halo_data_db = "halo_data"
        self.wings_config_db = "wings_config"
        
        # Cache for client configurations
        self._client_config_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("OrderContextService initialised")
    
    def get_order_context(
        self,
        orderref: str,
        client_ref: str,
        ddi: Optional[str] = None
    ) -> OrderContext:
        """
        Get order context for call analysis.
        
        Args:
            orderref: Order reference number
            client_ref: Client reference code
            ddi: DDI phone number (optional, for product matching)
            
        Returns:
            OrderContext with order and customer details
        """
        context = OrderContext(orderref=orderref)
        
        if not orderref or not client_ref:
            context.error = "Missing orderref or client_ref"
            return context
        
        try:
            with self.db.get_connection() as conn:
                # Step 1: Get table names for this client
                table_config = self._get_table_config(conn, client_ref)
                if not table_config:
                    logger.debug(f"No table configuration found for client {client_ref}")
                    context.error = "No table configuration for client"
                    return context
                
                orders_table = table_config.get("orders")
                customers_table = table_config.get("customers")
                products_table = table_config.get("products")
                
                if not orders_table:
                    logger.debug(f"No orders table configured for client {client_ref}")
                    context.error = "No orders table for client"
                    return context
                
                # Step 2: Check if tables exist
                if not self._table_exists(conn, self.halo_data_db, orders_table):
                    logger.debug(f"Orders table {orders_table} does not exist")
                    context.error = "Orders table not found"
                    return context
                
                # Step 3: Get client-specific field overrides
                field_mappings = self._get_field_mappings(conn, client_ref)
                
                # Step 4: Determine which columns exist in each table
                available_columns = self._get_available_columns(
                    conn, orders_table, customers_table, products_table
                )
                
                # Step 5: Build and execute query
                order_data = self._query_order_data(
                    conn,
                    orderref=orderref,
                    orders_table=orders_table,
                    customers_table=customers_table,
                    products_table=products_table,
                    ddi=ddi,
                    field_mappings=field_mappings,
                    available_columns=available_columns
                )
                
                if not order_data:
                    logger.debug(f"No order found for orderref {orderref}")
                    context.error = "Order not found"
                    return context
                
                # Step 6: Populate context from query results
                context = self._populate_context(context, order_data)
                
                logger.info(f"Order context loaded for {orderref}")
                return context
                
        except MySQLError as e:
            logger.error(f"Database error in get_order_context: {e}")
            context.error = f"Database error: {str(e)}"
            return context
        except Exception as e:
            logger.error(f"Error in get_order_context: {e}", exc_info=True)
            context.error = f"Error: {str(e)}"
            return context
    
    def _get_table_config(self, conn, client_ref: str) -> Optional[Dict[str, str]]:
        """
        Get table names for a client from halo_config.client_tables.
        
        Returns dict with keys: orders, customers, products, medias, calltypes
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(f"""
                SELECT orders, customers, products, medias, calltypes
                FROM {self.halo_config_db}.client_tables 
                WHERE clientref = %s
            """, (client_ref,))
            
            row = cursor.fetchone()
            cursor.close()
            
            return row if row else None
            
        except MySQLError as e:
            logger.warning(f"Error getting table config: {e}")
            return None
    
    def _table_exists(self, conn, database: str, table: str) -> bool:
        """Check if a table exists in the specified database."""
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (database, table))
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        except MySQLError:
            return False
    
    def _get_field_mappings(self, conn, client_ref: str) -> Dict[str, Dict[str, str]]:
        """
        Get field mappings, merging defaults with client-specific overrides.
        
        Queries wings_config.client_configuration for overrides.
        """
        mappings = DEFAULT_FIELD_MAPPINGS.copy()
        
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Check if table exists
            cursor.execute(f"""
                SELECT 1 FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'client_configuration'
            """, (self.wings_config_db,))
            
            if not cursor.fetchone():
                cursor.close()
                return mappings
            
            cursor.execute(f"""
                SELECT configuration 
                FROM {self.wings_config_db}.client_configuration 
                WHERE client_ref = %s
            """, (client_ref,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row and row.get("configuration"):
                try:
                    config = json.loads(row["configuration"])
                    
                    # Apply field_headers overrides
                    if "field_headers" in config:
                        for output_field, source_field in config["field_headers"].items():
                            if output_field in mappings and isinstance(source_field, str):
                                mappings[output_field]["field"] = source_field
                    
                    # Apply other custom field configurations
                    for key, value in config.items():
                        if key not in ["field_headers", "enquiry_order"] and isinstance(value, dict):
                            if "field" in value:
                                mappings[key] = {
                                    "table": value.get("table", "ords"),
                                    "field": value["field"],
                                    "expression": value.get("expression", "")
                                }
                                
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in client configuration for {client_ref}")
            
            return mappings
            
        except MySQLError as e:
            logger.warning(f"Error getting field mappings: {e}")
            return mappings
    
    def _get_available_columns(
        self,
        conn,
        orders_table: str,
        customers_table: Optional[str],
        products_table: Optional[str]
    ) -> Dict[str, set]:
        """
        Get available columns for each table to avoid querying non-existent fields.
        """
        available = {
            "ords": set(),
            "cust": set(),
            "prod": set(),
        }
        
        try:
            cursor = conn.cursor()
            
            # Orders table columns
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (self.halo_data_db, orders_table))
            available["ords"] = {row[0] for row in cursor.fetchall()}
            
            # Customers table columns
            if customers_table and self._table_exists(conn, self.halo_data_db, customers_table):
                cursor.execute(f"""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                """, (self.halo_data_db, customers_table))
                available["cust"] = {row[0] for row in cursor.fetchall()}
            
            # Products table columns
            if products_table and self._table_exists(conn, self.halo_config_db, products_table):
                cursor.execute(f"""
                    SELECT COLUMN_NAME 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                """, (self.halo_config_db, products_table))
                available["prod"] = {row[0] for row in cursor.fetchall()}
            
            cursor.close()
            
        except MySQLError as e:
            logger.warning(f"Error getting available columns: {e}")
        
        return available
    
    def _query_order_data(
        self,
        conn,
        orderref: str,
        orders_table: str,
        customers_table: Optional[str],
        products_table: Optional[str],
        ddi: Optional[str],
        field_mappings: Dict[str, Dict[str, str]],
        available_columns: Dict[str, set]
    ) -> Optional[Dict[str, Any]]:
        """
        Query order data with dynamic field selection based on available columns.
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Build SELECT clause with available fields only
            select_parts = []
            
            for output_name, mapping in field_mappings.items():
                table_alias = mapping["table"]
                field_name = mapping["field"]
                expression = mapping.get("expression", "")
                
                # Skip if table not available or field not in table
                if table_alias and table_alias not in available_columns:
                    continue
                if table_alias and field_name not in available_columns.get(table_alias, set()):
                    continue
                
                # Build field reference
                if table_alias:
                    field_ref = f"{table_alias}.`{field_name}`"
                else:
                    field_ref = field_name  # Literal value like "'order'"
                
                # Apply expression if present
                if expression:
                    field_expr = expression.replace("**FIELD**", field_ref)
                else:
                    field_expr = field_ref
                
                select_parts.append(f"{field_expr} AS `{output_name}`")
            
            if not select_parts:
                logger.warning("No valid fields to select")
                cursor.close()
                return None
            
            select_clause = ", ".join(select_parts)
            
            # Build FROM clause with JOINs
            from_clause = f"{self.halo_data_db}.`{orders_table}` AS ords"
            
            # Join customers table
            if customers_table and available_columns.get("cust"):
                from_clause += f"""
                    LEFT JOIN {self.halo_data_db}.`{customers_table}` AS cust 
                    ON ords.orderref = cust.orderref
                """
            
            # Join DDI table (for product matching via group/media)
            from_clause += f"""
                LEFT JOIN {self.halo_config_db}.ddi AS d 
                ON ords.ddi = d.ddi
            """
            
            # Join products table if available
            if products_table and available_columns.get("prod"):
                # Need to determine product field name
                product_field = field_mappings.get("product", {}).get("field", "product")
                if product_field in available_columns.get("ords", set()):
                    from_clause += f"""
                        LEFT JOIN {self.halo_config_db}.`{products_table}` AS prod 
                        ON prod.code = ords.`{product_field}`
                    """
            
            # Build and execute query
            query = f"""
                SELECT {select_clause}
                FROM {from_clause}
                WHERE ords.orderref = %s
                LIMIT 1
            """
            
            cursor.execute(query, (orderref,))
            row = cursor.fetchone()
            cursor.close()
            
            return row
            
        except MySQLError as e:
            logger.error(f"Error querying order data: {e}")
            return None
    
    def _populate_context(self, context: OrderContext, data: Dict[str, Any]) -> OrderContext:
        """
        Populate OrderContext from query results.
        """
        # Build customer name
        name_parts = []
        if data.get("title"):
            name_parts.append(str(data["title"]).strip())
        if data.get("fname"):
            name_parts.append(str(data["fname"]).strip())
        if data.get("surname"):
            name_parts.append(str(data["surname"]).strip())
        
        if name_parts:
            context.customer_name = " ".join(name_parts)
        
        # Build address
        address_parts = []
        for field in ["address1", "address2", "address3", "town", "county", "postcode"]:
            value = data.get(field)
            if value and str(value).strip():
                address_parts.append(str(value).strip())
        
        if address_parts:
            context.customer_address = ", ".join(address_parts)
        
        # Contact details
        if data.get("telephone"):
            context.customer_telephone = str(data["telephone"]).strip()
        if data.get("email_address"):
            context.customer_email = str(data["email_address"]).strip()
        
        # Order details
        if data.get("product"):
            context.product = str(data["product"]).strip()
        if data.get("product_desc"):
            context.product_description = str(data["product_desc"]).strip()
        
        if data.get("amount") is not None:
            try:
                context.amount = float(data["amount"])
            except (ValueError, TypeError):
                pass
        
        # Gift aid
        if data.get("giftaid"):
            context.giftaid = str(data["giftaid"]).strip()
        
        # Opt-ins
        if data.get("opt_post"):
            context.opt_post = str(data["opt_post"]).strip()
        if data.get("opt_phone"):
            context.opt_phone = str(data["opt_phone"]).strip()
        if data.get("opt_email"):
            context.opt_email = str(data["opt_email"]).strip()
        if data.get("opt_sms"):
            context.opt_sms = str(data["opt_sms"]).strip()
        
        return context


# Singleton instance
_order_context_service: Optional[OrderContextService] = None


def get_order_context_service() -> OrderContextService:
    """Get or create the singleton OrderContextService instance."""
    global _order_context_service
    if _order_context_service is None:
        _order_context_service = OrderContextService()
    return _order_context_service
