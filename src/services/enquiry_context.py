"""
Angel Intelligence - Enquiry Context Service

Retrieves enquiry data and available calltypes to provide context for call analysis.
Used to validate whether calls were logged under the correct calltype.

Database flow:
1. halo_config.client_tables → get calltypes table name for client
2. halo_config.ddi → get group code for DDI
3. halo_config.{calltypes_table} → get available calltypes filtered by group
4. halo_data.enquiry → get actual logged calltype
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import mysql.connector
from mysql.connector import Error as MySQLError

from src.config import get_settings
from src.database import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class CallTypeOption:
    """A single calltype option available for logging."""
    calltype: str  # Shorthand code
    description: str  # Full description
    
    def __str__(self) -> str:
        return f"{self.calltype}: {self.description}"


@dataclass
class EnquiryContext:
    """Context data for an enquiry, used in analysis prompts."""
    enqref: str
    logged_calltype: Optional[str] = None  # Shorthand code that was logged
    logged_calltype_description: Optional[str] = None  # Full description of logged calltype
    available_calltypes: Optional[List[CallTypeOption]] = None  # All valid options
    error: Optional[str] = None  # Any error encountered during lookup
    
    @property
    def has_context(self) -> bool:
        """Check if we have enough context for validation."""
        return self.logged_calltype is not None and self.available_calltypes is not None
    
    def to_prompt_context(self) -> str:
        """
        Format enquiry context for inclusion in analysis prompt.
        
        Returns empty string if insufficient context available.
        """
        if not self.has_context:
            return ""
        
        lines = [
            "CALL LOGGING VALIDATION:",
            f"- Enquiry Reference: {self.enqref}",
            f"- Logged Call Type: {self.logged_calltype_description or self.logged_calltype}",
            "",
            "Available Call Type Options (agent should have selected from these):",
        ]
        
        for ct in self.available_calltypes:
            lines.append(f"  • {ct.description}")
        
        lines.extend([
            "",
            "VALIDATION TASK:",
            "Based on the call content, determine if the agent selected the correct call type.",
            "If the logged call type does NOT match the call content, add a NEGATIVE score_impact",
            "with category 'Call_logging' and explain which call type would have been more appropriate.",
        ])
        
        return "\n".join(lines)


class EnquiryContextService:
    """
    Service to retrieve enquiry context for call analysis.
    
    Provides calltype validation context by:
    1. Looking up the client's calltypes table
    2. Finding the DDI group for filtering
    3. Retrieving available calltypes
    4. Getting the actual logged calltype from the enquiry
    """
    
    def __init__(self):
        """Initialise the enquiry context service."""
        self.settings = get_settings()
        self.db = DatabaseConnection()
        
        # Database names (same server, different databases)
        self.halo_config_db = "halo_config"
        self.halo_data_db = "halo_data"
        
        logger.info("EnquiryContextService initialised")
    
    def get_enquiry_context(
        self,
        enqref: str,
        client_ref: str,
        ddi: Optional[str] = None
    ) -> EnquiryContext:
        """
        Get enquiry context for call analysis.
        
        Args:
            enqref: Enquiry reference (URN in halo_data.enquiry)
            client_ref: Client reference code
            ddi: DDI phone number (optional, for group filtering)
            
        Returns:
            EnquiryContext with logged calltype and available options
        """
        context = EnquiryContext(enqref=enqref)
        
        if not enqref or not client_ref:
            context.error = "Missing enqref or client_ref"
            return context
        
        try:
            with self.db.get_connection() as conn:
                # Step 1: Get calltypes table name for this client
                calltypes_table = self._get_calltypes_table(conn, client_ref)
                if not calltypes_table:
                    logger.debug(f"No calltypes table found for client {client_ref}")
                    context.error = "No calltypes configuration for client"
                    return context
                
                # Step 2: Get group code for DDI (if provided)
                group_code = None
                if ddi:
                    group_code = self._get_ddi_group(conn, ddi)
                    logger.debug(f"DDI {ddi} maps to group: {group_code}")
                
                # Step 3: Get available calltypes
                available = self._get_available_calltypes(conn, calltypes_table, group_code)
                if available:
                    context.available_calltypes = available
                    logger.debug(f"Found {len(available)} available calltypes")
                else:
                    logger.debug(f"No calltypes found in {calltypes_table}")
                    context.error = "No calltypes available"
                    return context
                
                # Step 4: Get the actual logged calltype from enquiry
                logged = self._get_logged_calltype(conn, enqref)
                if logged:
                    context.logged_calltype = logged
                    # Find the description from available calltypes
                    for ct in available:
                        if ct.calltype == logged:
                            context.logged_calltype_description = ct.description
                            break
                    
                    # If not found in available list, the calltype might be from a different group
                    if not context.logged_calltype_description:
                        context.logged_calltype_description = self._lookup_calltype_description(
                            conn, calltypes_table, logged
                        )
                else:
                    logger.debug(f"No logged calltype found for enquiry {enqref}")
                
                return context
                
        except MySQLError as e:
            logger.error(f"Database error in get_enquiry_context: {e}")
            context.error = f"Database error: {str(e)}"
            return context
        except Exception as e:
            logger.error(f"Error in get_enquiry_context: {e}", exc_info=True)
            context.error = f"Error: {str(e)}"
            return context
    
    def _get_calltypes_table(self, conn, client_ref: str) -> Optional[str]:
        """
        Get the calltypes table name for a client.
        
        Queries: halo_config.client_tables WHERE clientref = {client_ref}
        Returns: calltypes field value, or None if not found/no field
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            # First check if the calltypes column exists in client_tables
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'client_tables' 
                AND COLUMN_NAME = 'calltypes'
            """, (self.halo_config_db,))
            
            if not cursor.fetchone():
                logger.debug("client_tables does not have calltypes column")
                cursor.close()
                return None
            
            # Get the calltypes table name
            cursor.execute(f"""
                SELECT calltypes 
                FROM {self.halo_config_db}.client_tables 
                WHERE clientref = %s
            """, (client_ref,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row and row.get("calltypes"):
                return row["calltypes"]
            
            return None
            
        except MySQLError as e:
            logger.warning(f"Error getting calltypes table: {e}")
            return None
    
    def _get_ddi_group(self, conn, ddi: str) -> Optional[str]:
        """
        Get the group code for a DDI.
        
        Queries: halo_config.ddi WHERE ddi = {ddi}
        Returns: group field value, or None if not found
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Check if group column exists in ddi table
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'ddi' 
                AND COLUMN_NAME = 'group'
            """, (self.halo_config_db,))
            
            if not cursor.fetchone():
                logger.debug("ddi table does not have group column")
                cursor.close()
                return None
            
            cursor.execute(f"""
                SELECT `group` 
                FROM {self.halo_config_db}.ddi 
                WHERE ddi = %s
            """, (ddi,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return row.get("group")
            
            return None
            
        except MySQLError as e:
            logger.warning(f"Error getting DDI group: {e}")
            return None
    
    def _get_available_calltypes(
        self, 
        conn, 
        calltypes_table: str, 
        group_code: Optional[str]
    ) -> Optional[List[CallTypeOption]]:
        """
        Get available calltypes from the client's calltypes table.
        
        Queries: halo_config.{calltypes_table}
        Fallback chain when group_code provided:
        1. Exact match: WHERE group = {group_code}
        2. Pattern match: WHERE group = {first_char}? (e.g., 'A1' -> 'A?')
        3. Ungrouped: WHERE group IS NULL
        
        If no group column exists: returns all calltypes
        
        Returns: List of CallTypeOption, or None if table doesn't exist
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Verify the table exists
            cursor.execute(f"""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s
            """, (self.halo_config_db, calltypes_table))
            
            if not cursor.fetchone():
                logger.warning(f"Calltypes table {calltypes_table} does not exist")
                cursor.close()
                return None
            
            # Check if required columns exist
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s 
                AND COLUMN_NAME IN ('calltype', 'desc')
            """, (self.halo_config_db, calltypes_table))
            
            columns = {row["COLUMN_NAME"] for row in cursor.fetchall()}
            if "calltype" not in columns or "desc" not in columns:
                logger.warning(f"Calltypes table {calltypes_table} missing required columns")
                cursor.close()
                return None
            
            # Check if group column exists
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = %s 
                AND COLUMN_NAME = 'group'
            """, (self.halo_config_db, calltypes_table))
            
            has_group_column = cursor.fetchone() is not None
            
            # Build query based on group availability
            if has_group_column and group_code:
                # Step 1: Try exact group match
                cursor.execute(f"""
                    SELECT calltype, `desc` 
                    FROM {self.halo_config_db}.`{calltypes_table}` 
                    WHERE `group` = %s
                    ORDER BY `desc`
                """, (group_code,))
                
                rows = cursor.fetchall()
                
                # Step 2: If no exact match, try pattern match (first char + '?')
                if not rows and len(group_code) >= 1:
                    pattern_group = group_code[0] + "?"
                    logger.debug(f"No calltypes for group {group_code}, trying pattern {pattern_group}")
                    cursor.execute(f"""
                        SELECT calltype, `desc` 
                        FROM {self.halo_config_db}.`{calltypes_table}` 
                        WHERE `group` = %s
                        ORDER BY `desc`
                    """, (pattern_group,))
                    rows = cursor.fetchall()
                
                # Step 3: If still no matches, fall back to NULL group (global calltypes)
                if not rows:
                    logger.debug(f"No calltypes for group {group_code} or pattern, falling back to global")
                    cursor.execute(f"""
                        SELECT calltype, `desc` 
                        FROM {self.halo_config_db}.`{calltypes_table}` 
                        WHERE `group` IS NULL
                        ORDER BY `desc`
                    """)
                    rows = cursor.fetchall()
            elif has_group_column:
                # No group code, get global calltypes (NULL group)
                cursor.execute(f"""
                    SELECT calltype, `desc` 
                    FROM {self.halo_config_db}.`{calltypes_table}` 
                    WHERE `group` IS NULL
                    ORDER BY `desc`
                """)
                rows = cursor.fetchall()
            else:
                # No group column, get all calltypes
                cursor.execute(f"""
                    SELECT calltype, `desc` 
                    FROM {self.halo_config_db}.`{calltypes_table}`
                    ORDER BY `desc`
                """)
                rows = cursor.fetchall()
            
            cursor.close()
            
            if not rows:
                return None
            
            return [
                CallTypeOption(
                    calltype=row["calltype"],
                    description=row["desc"] or row["calltype"]
                )
                for row in rows
            ]
            
        except MySQLError as e:
            logger.warning(f"Error getting available calltypes: {e}")
            return None
    
    def _get_logged_calltype(self, conn, enqref: str) -> Optional[str]:
        """
        Get the calltype that was logged on the enquiry.
        
        Queries: halo_data.enquiry WHERE URN = {enqref}
        Returns: calltype field value
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            # Check if calltype column exists
            cursor.execute(f"""
                SELECT COLUMN_NAME 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = %s 
                AND TABLE_NAME = 'enquiry' 
                AND COLUMN_NAME = 'calltype'
            """, (self.halo_data_db,))
            
            if not cursor.fetchone():
                logger.debug("enquiry table does not have calltype column")
                cursor.close()
                return None
            
            cursor.execute(f"""
                SELECT calltype 
                FROM {self.halo_data_db}.enquiry 
                WHERE URN = %s
            """, (enqref,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return row.get("calltype")
            
            return None
            
        except MySQLError as e:
            logger.warning(f"Error getting logged calltype: {e}")
            return None
    
    def _lookup_calltype_description(
        self, 
        conn, 
        calltypes_table: str, 
        calltype: str
    ) -> Optional[str]:
        """
        Look up the description for a calltype code.
        
        Used when the logged calltype isn't in the filtered available list
        (e.g., logged under a different group's calltype).
        """
        try:
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(f"""
                SELECT `desc` 
                FROM {self.halo_config_db}.`{calltypes_table}` 
                WHERE calltype = %s
            """, (calltype,))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return row.get("desc")
            
            return None
            
        except MySQLError as e:
            logger.warning(f"Error looking up calltype description: {e}")
            return None


# Singleton instance
_enquiry_context_service: Optional[EnquiryContextService] = None


def get_enquiry_context_service() -> EnquiryContextService:
    """Get or create the singleton EnquiryContextService instance."""
    global _enquiry_context_service
    if _enquiry_context_service is None:
        _enquiry_context_service = EnquiryContextService()
    return _enquiry_context_service
