"""
Admin CLI to manage portal users.

Usage:
    python portal/manage_users.py list
    python portal/manage_users.py add subscriber@email.com "John Smith" password123
    python portal/manage_users.py deactivate subscriber@email.com
    python portal/manage_users.py reset-password subscriber@email.com newpassword
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from portal.users import init_db, create_user, list_users, deactivate_user, change_password

init_db()

def cmd_list():
    users = list_users()
    if not users:
        print("No users found.")
        return
    print(f"\n{'Email':<35} {'Name':<20} {'Role':<12} {'Active':<8} {'Created'}")
    print("-" * 95)
    for u in users:
        active = "Yes" if u["active"] else "No"
        print(f"{u['email']:<35} {u['name']:<20} {u['role']:<12} {active:<8} {u['created_at'][:10]}")
    print()

def cmd_add(email, name, password, role="subscriber"):
    try:
        create_user(email, password, name, role)
        print(f"Created user: {email} ({role})")
        print(f"They can log in at the portal with:")
        print(f"  Email:    {email}")
        print(f"  Password: {password}")
    except ValueError as e:
        print(f"Error: {e}")

def cmd_deactivate(email):
    deactivate_user(email)
    print(f"Deactivated: {email}")

def cmd_reset(email, new_password):
    change_password(email, new_password)
    print(f"Password reset for: {email}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    if cmd == "list":
        cmd_list()
    elif cmd == "add" and len(args) >= 4:
        role = args[4] if len(args) > 4 else "subscriber"
        cmd_add(args[1], args[2], args[3], role)
    elif cmd == "deactivate" and len(args) >= 2:
        cmd_deactivate(args[1])
    elif cmd == "reset-password" and len(args) >= 3:
        cmd_reset(args[1], args[2])
    else:
        print(__doc__)