\set ON_ERROR_STOP on
-- psql quotes identifiers and literals separately; never interpolate raw SQL.
\getenv owner_user OWNER_DB_USER
\getenv owner_password OWNER_DB_PASSWORD
\getenv app_user APP_DB_USER
\getenv app_password APP_DB_PASSWORD
\getenv ro_user RO_DB_USER
\getenv ro_password RO_DB_PASSWORD
\getenv db_name POSTGRES_DB

BEGIN;
CREATE ROLE :"owner_user" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'owner_password';
CREATE ROLE :"app_user" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'app_password';
CREATE ROLE :"ro_user" LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
    PASSWORD :'ro_password';

ALTER DATABASE :"db_name" OWNER TO :"owner_user";
-- PUBLIC otherwise grants TEMPORARY to every login, including chatbot_ro.
REVOKE ALL ON DATABASE :"db_name" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"db_name" TO :"app_user", :"ro_user";
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE EXTENSION IF NOT EXISTS vector;
ALTER ROLE :"ro_user" SET statement_timeout = '5s';
-- Object-level grants arrive with the migrations that introduce those objects.
COMMIT;
