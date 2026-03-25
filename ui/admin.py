import streamlit as st
from auth.credentials import list_users, add_user, remove_user, change_password


def show_admin():
    user = st.session_state.get("auth_user", {})
    if user.get("role") != "superadmin":
        st.error("Доступ запрещён.")
        return

    st.markdown("""
<style>
[data-testid="stMetricValue"] > div { font-size: 28px !important; color: #0071E3 !important; }
</style>
""", unsafe_allow_html=True)

    st.markdown("## ⚙️ Управление пользователями")
    st.caption("Только для суперадмина · weerowoolf")
    st.markdown("---")

    users = list_users()
    c1, c2, c3 = st.columns(3)
    c1.metric("Всего пользователей", len(users))
    c2.metric("Демо-аккаунтов", sum(1 for u in users if u["role"] == "demo"))
    c3.metric("Суперадминов", sum(1 for u in users if u["role"] == "superadmin"))

    st.markdown("### 👥 Список пользователей")
    st.markdown("""
<style>
.user-row { display:flex; align-items:center; gap:14px; padding:10px 16px;
            background:#FFFFFF; border-radius:12px; margin-bottom:6px;
            box-shadow:0 1px 8px rgba(0,0,0,0.06); }
.user-name { font-weight:600; color:#1D1D1F; font-size:14px; flex:1; }
.user-role { font-size:11px; font-weight:700; padding:2px 10px;
             border-radius:980px; letter-spacing:0.05em; }
.role-superadmin { background:#0071E3; color:#fff; }
.role-demo       { background:#F5F5F7; color:#6E6E73; border:1px solid #E5E5EA; }
.user-date { font-size:12px; color:#AEAEB2; }
</style>
""", unsafe_allow_html=True)

    for u in users:
        role_cls = "role-superadmin" if u["role"] == "superadmin" else "role-demo"
        st.markdown(f"""
<div class="user-row">
  <span class="user-name">@{u['username']}</span>
  <span style="color:#6E6E73;font-size:13px;">{u['name']}</span>
  <span class="user-role {role_cls}">{u['role'].upper()}</span>
  <span class="user-date">{u['created_at']}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    col_add, col_del = st.columns(2)

    with col_add:
        st.markdown("### ➕ Добавить пользователя")
        with st.form("add_user_form"):
            new_user = st.text_input("Логин", placeholder="username", key="nu_login")
            new_pass = st.text_input("Пароль", type="password", placeholder="Пароль", key="nu_pass")
            new_name = st.text_input("Имя (для отображения)", placeholder="Имя клиента", key="nu_name")
            new_role = st.selectbox("Роль", ["demo", "superadmin"], key="nu_role")
            if st.form_submit_button("Создать аккаунт", use_container_width=True):
                if add_user(new_user, new_pass, new_name, new_role):
                    st.success(f"Пользователь @{new_user} создан!")
                    st.rerun()
                else:
                    st.error("Ошибка — проверьте логин и пароль.")

    with col_del:
        st.markdown("### 🗑️ Удалить / изменить пароль")
        user_logins = [u["username"] for u in users if u["username"] != "weerowoolf"]
        if user_logins:
            target = st.selectbox("Пользователь", user_logins, key="del_target")

            with st.form("del_form"):
                if st.form_submit_button("🗑️ Удалить пользователя", use_container_width=True):
                    if remove_user(target):
                        st.success(f"@{target} удалён.")
                        st.rerun()
                    else:
                        st.error("Нельзя удалить суперадмина.")

            with st.form("chpw_form"):
                new_pw = st.text_input("Новый пароль", type="password", key="chpw_new")
                if st.form_submit_button("🔑 Изменить пароль", use_container_width=True):
                    if change_password(target, new_pw):
                        st.success("Пароль изменён.")
                    else:
                        st.error("Ошибка.")
        else:
            st.info("Нет дополнительных пользователей.")

    st.markdown("---")
    st.markdown("### 🔗 Ссылка на приложение")
    st.info("Поделитесь этой ссылкой с клиентом — он увидит лендинг и форму входа.\nСоздайте для него отдельный аккаунт и отправьте логин + пароль в Telegram.")
