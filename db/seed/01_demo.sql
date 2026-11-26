-- Synthetic fixtures only. Fixed IDs and ON CONFLICT make repeated loads safe.
BEGIN;

INSERT INTO raw.clients (id, name, contact_email, internal_notes) VALUES
    (1, 'Northstar Labs', 'demo@northstar.example', 'Synthetic private client notes'),
    (2, 'Harbor Logistics', 'demo@harbor.example', 'Synthetic private client notes'),
    (3, 'Cedar Retail', 'demo@cedar.example', 'Synthetic private client notes'),
    (4, 'Orion Energy', 'demo@orion.example', 'Synthetic private client notes')
ON CONFLICT (id) DO NOTHING;

INSERT INTO raw.projects
    (id, client_id, name, description, status, start_date, end_date, budget_usd, internal_notes)
VALUES
    (101, 1, 'Research Portal', 'Searchable workspace for research reports.',
     'active', '2026-01-15', '2026-12-31', 120000, 'Synthetic confidential pricing'),
    (102, 1, 'Lab Inventory', 'Track laboratory equipment and supplies.',
     'completed', '2025-06-01', '2026-02-28', 80000, 'Synthetic confidential pricing'),
    (103, 1, 'Experiment Planner', 'Plan experiments and review milestones.',
     'planned', '2027-01-10', NULL, 65000, 'Synthetic confidential pricing'),
    (104, 2, 'Fleet Dashboard', 'Operational dashboard for the delivery fleet.',
     'active', '2026-03-01', NULL, 180000, 'Synthetic confidential pricing'),
    (105, 2, 'Route Optimizer', 'Optimize delivery routes and reduce fuel use.',
     'on_hold', '2026-05-01', '2027-03-31', 210000, 'Synthetic confidential pricing'),
    (106, 2, 'Warehouse Scanner', 'Barcode scanning for warehouse operations.',
     'completed', '2025-02-01', '2025-11-30', 95000, 'Synthetic confidential pricing'),
    (107, 3, 'Customer Portal', 'Customer accounts and order tracking.',
     'active', '2026-02-10', '2026-11-30', 140000, 'Synthetic confidential pricing'),
    (108, 3, 'Stock Forecast', 'Forecast seasonal inventory demand.',
     'active', '2026-07-01', NULL, 110000, 'Synthetic confidential pricing'),
    (109, 3, 'Loyalty Refresh', 'Refresh the retail loyalty experience.',
     'cancelled', '2026-04-01', '2026-08-15', 45000, 'Synthetic confidential pricing'),
    (110, 4, 'Grid Monitor', 'Monitor energy generation and grid health.',
     'active', '2026-01-05', '2027-06-30', 260000, 'Synthetic confidential pricing'),
    (111, 4, 'Solar Analytics', 'Analyze solar production and maintenance.',
     'planned', '2027-02-01', '2027-12-31', 190000, 'Synthetic confidential pricing'),
    (112, 4, 'Meter Migration', 'Migrate legacy meter readings.',
     'completed', '2025-09-01', '2026-06-30', 75000, 'Synthetic confidential pricing')
ON CONFLICT (id) DO NOTHING;

COMMIT;
