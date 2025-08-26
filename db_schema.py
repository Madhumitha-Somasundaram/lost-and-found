from sqlalchemy import text
from database_connection import engine

def create_lost_items_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS lost_items (
                id INT PRIMARY KEY AUTO_INCREMENT,
                item_type VARCHAR(255),
                brand VARCHAR(255),
                color VARCHAR(255),
                item_condition VARCHAR(255),
                description TEXT,
                hidden_details TEXT,
                found_location VARCHAR(255),
                pickup_location VARCHAR(255),
                image_url VARCHAR(500),
                uploader_name VARCHAR(50),
                uploader_email VARCHAR(50),
                status ENUM('unclaimed', 'claimed') DEFAULT 'unclaimed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
def create_unclaimed_items_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS unclaimed_items (
                id INT PRIMARY KEY AUTO_INCREMENT,
                uploader_name VARCHAR(50),
                uploader_email VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))



def initialize_tables():
    create_lost_items_table()
    create_unclaimed_items_table()
