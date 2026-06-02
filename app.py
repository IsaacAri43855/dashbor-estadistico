import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURACIÓN GENERAL DEL SISTEMA Y DISEÑO VISUAL
# ==============================================================================
st.set_page_config(
    page_title="Analítica Estadística Descriptiva",
    page_icon="📊",
    layout="wide"
)

# Configuración estética global para los gráficos (Matplotlib & Seaborn)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#E2E8F0',
    'axes.linewidth': 1.0,
    'grid.color': '#F1F5F9',
    'xtick.color': '#64748B',
    'ytick.color': '#64748B',
    'text.color': '#1E293B',
    'axes.labelcolor': '#475569',
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'figure.titlesize': 16
})

# ==============================================================================
# FUNCIONES AUXILIARES DE LIMPIEZA DE DATOS
# ==============================================================================
def limpiar_columna_numerica(series):
    """Limpia caracteres especiales comunes (% o $) y convierte de texto a número."""
    try:
        s = series.astype(str).str.replace('%', '', regex=False)
        s = s.str.replace('$', '', regex=False)
        s = s.str.replace(',', '.', regex=False)  # Cambia comas decimales por puntos
        s = s.str.strip()
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.to_numeric(series, errors='coerce')

# ==============================================================================
# INYECTOR DE DATOS MULTIPROPÓSITO (DATASET DE PRUEBA)
# ==============================================================================
@st.cache_data
def generar_dataset_predeterminado():
    """Genera un dataset sintético de rendimiento operativo y de clientes."""
    np.random.seed(101)
    n = 150
    data = {
        "Canal_Adquisicion": np.random.choice(["Web", "App Móvil", "Social Media", "Recomendación"], size=n, p=[0.4, 0.3, 0.15, 0.15]),
        "Transacciones_Mes": np.random.randint(1, 16, size=n),
        "Ticket_Medio_USD": np.round(np.random.normal(120, 35, size=n), 2),
        "Satisfaccion_Cliente": np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.1, 0.2, 0.45, 0.2]),
        "Soporte_Requerido": np.random.choice(["Bajo", "Medio", "Alto"], size=n, p=[0.6, 0.3, 0.1]),
        "Edad_Usuario": np.random.randint(18, 65, size=n)
    }
    return pd.DataFrame(data)

# ==============================================================================
# ESTRUCTURACIÓN DE LA INTERFAZ DE USUARIO (DASHBOARD CORE)
# ==============================================================================
st.title("📈 Plataforma Integral de Analítica Estadística")
st.markdown("Procesamiento, distribución de frecuencias y análisis descriptivo automatizado para variables cualitativas y cuantitativas.")
st.markdown("---")

# Panel lateral: Carga dinámica y mapeo
st.sidebar.header("📁 Origen de Datos")
archivo_cargado = st.sidebar.file_uploader(
    "Cargar base de datos (Formatos soportados: CSV, XLSX):", 
    type=["csv", "xlsx"]
)

# Validación y lectura del archivo con DETECTOR AUTOMÁTICO DE SEPARADORES
if archivo_cargado is not None:
    try:
        if archivo_cargado.name.endswith('.csv'):
            # Detectar el delimitador automáticamente leyendo los primeros bytes
            sample = archivo_cargado.read(2048)
            archivo_cargado.seek(0) # Volver al inicio del archivo
            
            try:
                sample_str = sample.decode('utf-8')
            except UnicodeDecodeError:
                sample_str = sample.decode('latin-1')
                
            # Conteo de separadores comunes para identificar la estructura del CSV
            semi_count = sample_str.count(';')
            comma_count = sample_str.count(',')
            tab_count = sample_str.count('\t')
            
            if semi_count > comma_count and semi_count > tab_count:
                separador = ';'
            elif tab_count > comma_count and tab_count > semi_count:
                separador = '\t'
            else:
                separador = ','
                
            df = pd.read_csv(archivo_cargado, sep=separador)
        else:
            df = pd.read_excel(archivo_cargado)
        
        st.sidebar.success("¡Datos cargados con éxito!")
    except Exception as e:
        st.sidebar.error(f"Error al procesar el archivo: {e}")
        df = generar_dataset_predeterminado()
else:
    df = generar_dataset_predeterminado()
    st.sidebar.info("💡 Visualizando base de datos predeterminada de rendimiento y clientes.")

# Limpieza inicial de nombres de columnas para evitar espacios vacíos molestos
df.columns = [col.strip() for col in df.columns]
# Filtrar columnas completamente vacías generadas por error
df = df.dropna(how='all', axis=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Identificación del tipo de columnas
columnas_categoricas = df.select_dtypes(include=[object, "category"]).columns.tolist()
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

# Si Pandas detectó números como texto debido a formatos raros (como "7%"), nos aseguramos de dar la opción
for col in df.columns:
    if col not in columnas_numericas and col not in columnas_categoricas:
        columnas_categoricas.append(col)

# Respaldos de contingencia si el dataset está vacío
if not columnas_categoricas:
    columnas_categoricas = df.columns.tolist()
if not columnas_numericas:
    columnas_numericas = df.columns.tolist()

# Panel lateral: Mapeo de Variables Estadísticas
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Configuración de Variables")

var_cualitativa = st.sidebar.selectbox(
    "Variable Cualitativa (Categorías/Texto):",
    columnas_categoricas,
    index=0
)

var_discreta = st.sidebar.selectbox(
    "Variable Cuantitativa Discreta (Números Enteros/Conteos):",
    df.columns.tolist(), # Permitimos mapear cualquier columna y la limpiaremos en caliente
    index=df.columns.tolist().index(columnas_numericas[0]) if columnas_numericas[0] in df.columns else 0
)

var_continua = st.sidebar.selectbox(
    "Variable Cuantitativa Continua (Agrupada):",
    df.columns.tolist(), # Permitimos mapear cualquier columna y la limpiaremos en caliente
    index=df.columns.tolist().index(columnas_numericas[-1]) if len(columnas_numericas) > 1 else 0
)

# Creación de vistas modulares mediante pestañas limpias
tab_explorador, tab_cualitativa, tab_discreta, tab_agrupada = st.tabs([
    "📋 Exploración de Datos",
    "🗣️ Análisis Cualitativo",
    "🔢 Análisis Discreto",
    "📐 Análisis Agrupado (Sturges)"
])

# ==============================================================================
# PESTAÑA 1: EXPLORACIÓN DE DATOS (DATA EXPLORER)
# ==============================================================================
with tab_explorador:
    st.header("📊 Estructura y Composición del Dataset")
    
    # Métricas descriptivas del sistema
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Registros", df.shape[0])
    m2.metric("Total Variables", df.shape[1])
    m3.metric("Variables Cualitativas", len(columnas_categoricas))
    m4.metric("Celdas Vacías (Nulos)", df.isnull().sum().sum())
    
    # Vista interactiva
    st.subheader("🔍 Visor Dinámico de la Tabla de Datos")
    st.dataframe(df, use_container_width=True)
    
    # Resumen rápido de tipos de datos
    st.subheader("⚙️ Atributos Detectados")
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.markdown("**Columnas de Texto / Categorías:**")
        st.write(columnas_categoricas)
    with col_der:
        st.markdown("**Columnas Numéricas:**")
        st.write(columnas_numericas)

# ==============================================================================
# PESTAÑA 2: ANÁLISIS CUALITATIVO (CATEGORICAL ANALYSIS)
# ==============================================================================
with tab_cualitativa:
    st.header("🗣️ Distribución de Frecuencias de Variables Cualitativas")
    st.markdown(f"Análisis enfocado en categorías nominales u ordinales de la variable: `{var_cualitativa}`")
    
    # Algoritmo de agregación estadística (Tabla de Frecuencias)
    frec_cualita = df[var_cualitativa].dropna().value_counts().reset_index()
    frec_cualita.columns = [var_cualitativa, "fi"]
    frec_cualita["hi"] = frec_cualita["fi"] / len(df)
    frec_cualita["hip"] = frec_cualita["hi"] * 100
    frec_cualita["Fi"] = frec_cualita["fi"].cumsum()
    frec_cualita["Hi"] = frec_cualita["hi"].cumsum()
    
    # Tabla formateada de salida
    st.write("**Estructura de Frecuencias Relativas y Acumuladas:**")
    st.dataframe(frec_cualita.style.format({
        "hi": "{:.4f}",
        "hip": "{:.2f}%",
        "Hi": "{:.4f}"
    }), use_container_width=True)
    
    # Renderizado gráfico
    st.subheader("📈 Representación Gráfica de Categorías")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Distribución de Frecuencia Absoluta**")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            x=frec_cualita[var_cualitativa].astype(str), 
            y=frec_cualita["fi"], 
            palette="Blues_r", 
            edgecolor="#CBD5E1", 
            ax=ax
        )
        ax.set_title(f"Volumen por {var_cualitativa.replace('_', ' ')}", pad=15)
        ax.set_xlabel(None)
        ax.set_ylabel("Frecuencia Absoluta (fi)")
        
        # Evitar amontonamiento si hay muchas categorías cualitativas
        if len(frec_cualita) > 10:
            plt.xticks(rotation=45, ha='right', fontsize=8)
            
        st.pyplot(fig)
        plt.close(fig)
        
    with c2:
        st.markdown("**Participación Porcentual Relativa**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(
            frec_cualita['hi'], 
            labels=frec_cualita[var_cualitativa].astype(str), 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("Pastel1"),
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        ax.set_title("Composición Relativa", pad=15)
        st.pyplot(fig)
        plt.close(fig)

# ==============================================================================
# PESTAÑA 3: ANÁLISIS DISCRETO (DISCRETE NUMERICAL ANALYSIS) - MEJORADA
# ==============================================================================
with tab_discreta:
    st.header("🔢 Distribución de Frecuencias de Variables Cuantitativas Discretas")
    st.markdown(f"Análisis enfocado en valores numéricos de la variable: `{var_discreta}`")
    
    # LIMPIEZA ROBUSTA: Limpia caracteres que no correspondan a floats
    serie_discreta_limpia = limpiar_columna_numerica(df[var_discreta])
    
    df_temp_discr = df.copy()
    df_temp_discr["_temp_discreta"] = serie_discreta_limpia
    df_temp_discr = df_temp_discr.dropna(subset=['_temp_discreta'])
    
    if len(df_temp_discr) == 0:
        st.error(f"No se pudieron extraer datos numéricos limpios de la columna `{var_discreta}`. Por favor selecciona otra variable en la barra lateral.")
    else:
        # Algoritmo de agregación y conteo ordenado
        tabla_discreta = df_temp_discr["_temp_discreta"].value_counts().sort_index().reset_index()
        tabla_discreta.columns = ["Valor_X", "fi"]
        tabla_discreta["hi"] = tabla_discreta["fi"] / len(df_temp_discr)
        tabla_discreta["hip"] = tabla_discreta["hi"] * 100
        tabla_discreta["Fi"] = tabla_discreta["fi"].cumsum()
        tabla_discreta["Hi"] = tabla_discreta["hi"].cumsum()
        
        # ALERTA DE ALTA CARDINALIDAD (Evita que el gráfico de bastones se vea como pared azul)
        cant_valores_unicos = len(tabla_discreta)
        if cant_valores_unicos > 25:
            st.warning(f"⚠️ **Sugerencia de visualización:** La variable `{var_discreta}` posee {cant_valores_unicos} valores únicos. Para conjuntos de datos tan amplios, te recomendamos revisar la pestaña **Análisis Agrupado (Sturges)** para verlos agrupados en rangos cómodos. No obstante, hemos optimizado este gráfico de bastones para ti:")
            
        st.write("**Estructura de Distribución de Frecuencias Discretas:**")
        st.dataframe(tabla_discreta.style.format({
            "Valor_X": "{:.2f}",
            "hi": "{:.4f}",
            "hip": "{:.2f}%",
            "Hi": "{:.4f}"
        }), use_container_width=True)
        
        # Visualización mediante Diagrama de Bastones Mejorado e Inteligente
        st.subheader("🖼️ Diagrama de Bastones (Lollipop Chart)")
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # --- AJUSTE DINÁMICO DE GROSOR Y TAMAÑO PARA EVITAR "PAREDES" ---
        if cant_valores_unicos > 100:
            grosor_linea = 0.4
            tamano_punto = 1.5
        elif cant_valores_unicos > 50:
            grosor_linea = 0.8
            tamano_punto = 3.0
        elif cant_valores_unicos > 25:
            grosor_linea = 1.2
            tamano_punto = 4.0
        else:
            grosor_linea = 2.0
            tamano_punto = 6.0
            
        # Dibujar bastones finos adaptativos
        ax.vlines(
            x=tabla_discreta['Valor_X'], 
            ymin=0, 
            ymax=tabla_discreta['fi'], 
            colors='#3B82F6', 
            linewidth=grosor_linea,
            alpha=0.7
        )
        
        # Puntos de frecuencia adaptativos
        ax.plot(
            tabla_discreta['Valor_X'], 
            tabla_discreta['fi'], 
            "o", 
            color='#EF4444', 
            markersize=tamano_punto,
            label="Frecuencia Absoluta"
        )
        
        # --- EVITAR AMONTONAMIENTO EN EL EJE X: Selección inteligente de marcas ---
        if cant_valores_unicos > 15:
            # Seleccionamos exactamente 15 marcas distribuidas a lo largo del eje X
            indices_a_mostrar = np.linspace(0, cant_valores_unicos - 1, num=15, dtype=int)
            ticks_a_mostrar = tabla_discreta['Valor_X'].iloc[indices_a_mostrar]
            ax.set_xticks(ticks_a_mostrar)
            # Rotar textos para que queden diagonales y legibles
            plt.xticks(rotation=45, ha='right', fontsize=9)
        else:
            # Si son pocos, mostrar todos con total normalidad
            ax.set_xticks(tabla_discreta['Valor_X'])
            
        ax.set_title(f"Distribución Detallada de {var_discreta.replace('_', ' ')}", pad=15)
        ax.set_xlabel("Valor Variable (X)")
        ax.set_ylabel("Frecuencia (fi)")
        st.pyplot(fig)
        plt.close(fig)

# ==============================================================================
# PESTAÑA 4: ANÁLISIS CUANTITATIVO AGRUPADO (STURGES CONTINUOUS)
# ==============================================================================
with tab_agrupada:
    st.header("📐 Distribución de Frecuencias Agrupadas")
    st.markdown(f"Segmentación de la variable `{var_continua}` mediante el método de discretización automática de Sturges.")
    
    # LIMPIEZA ROBUSTA: Limpia caracteres especiales para la variable continua
    serie_continua_limpia = limpiar_columna_numerica(df[var_continua])
    
    df_temp_cont = df.copy()
    df_temp_cont["_temp_continua"] = serie_continua_limpia
    df_temp_cont = df_temp_cont.dropna(subset=['_temp_continua'])
    
    if len(df_temp_cont) == 0:
        st.error(f"No se pudieron extraer datos numéricos limpios de la columna `{var_continua}`. Por favor selecciona otra variable en la barra lateral.")
    else:
        # --------------------------------------------------------------------------
        # NÚCLEO MATEMÁTICO: REGLA DE STURGES
        # --------------------------------------------------------------------------
        n = len(df_temp_cont)
        min_val = float(df_temp_cont["_temp_continua"].min())
        max_val = float(df_temp_cont["_temp_continua"].max())
        rango = max_val - min_val
        
        # Manejo de contingencia para rangos planos
        if rango == 0:
            rango = 1.0
            
        # Cálculo de intervalos mediante la Regla de Sturges: k = 1 + 3.322 * log10(n)
        k = int(np.ceil(1 + 3.322 * np.log10(n)))
        amplitud = rango / k
        
        st.markdown(f"""
        Para agrupar la información de `{var_continua}`, aplicamos la **Regla de Sturges** para obtener el número óptimo de intervalos ($k$) y la amplitud ($A$):
        * **Fórmula de Sturges:** $k = 1 + 3.322 \\log_{{10}}({n}) = {1 + 3.322 * np.log10(n):.4f} \\rightarrow$ Redondeado hacia arriba: **{k}**
        * **Amplitud de clase:** $A = \\frac{{R}}{{k}} = \\frac{{{rango:.2f}}}{{{k}}} = {amplitud:.4f}$
        """)
        
        # Mostrar parámetros críticos
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Muestra (n)", n)
        c_m2.metric("Rango (R)", f"{rango:.2f}")
        c_m3.metric("Clases / Intervalos (k)", k)
        c_m4.metric("Amplitud de Intervalo (A)", f"{amplitud:.4f}")
        
        # Generación de puntos de corte exactos
        cortes = np.arange(min_val, max_val + amplitud + 1e-5, amplitud)
        
        # Agrupamiento de datos (pd.cut)
        df_temp_cont["intervalos"] = pd.cut(df_temp_cont["_temp_continua"], bins=cortes, include_lowest=True, right=False)
        
        # Generación de la tabla de frecuencias para datos agrupados
        tabla_agrupada = df_temp_cont["intervalos"].value_counts().sort_index().reset_index()
        tabla_agrupada.columns = ["Intervalo", "fi"]
        tabla_agrupada["Intervalo"] = tabla_agrupada["Intervalo"].astype(str)
        
        # Extracción matemática de la marca de clase (Xi)
        def extraer_marca_clase(intervalo_texto):
            try:
                clean = intervalo_texto.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                lim_inf, lim_sup = map(float, clean.split(','))
                return (lim_inf + lim_sup) / 2
            except Exception:
                return 0.0
                
        tabla_agrupada["Xi (Marca de Clase)"] = tabla_agrupada["Intervalo"].apply(extraer_marca_clase)
        
        # Cálculos de porcentajes y frecuencias acumuladas
        tabla_agrupada["hi"] = tabla_agrupada["fi"] / n
        tabla_agrupada["hip"] = tabla_agrupada["hi"] * 100
        tabla_agrupada["Fi"] = tabla_agrupada["fi"].cumsum()
        tabla_agrupada["Hi"] = tabla_agrupada["hi"].cumsum()
        
        st.write("**Tabla de Frecuencias por Intervalo de Clase:**")
        st.dataframe(tabla_agrupada.style.format({
            "Xi (Marca de Clase)": "{:.2f}",
            "hi": "{:.4f}",
            "hip": "{:.2f}%",
            "Hi": "{:.4f}"
        }), use_container_width=True)
        
        # Visualizaciones avanzadas para datos continuos
        st.subheader("🖼️ Representaciones de Distribución Continua")
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("**Histograma y Polígono de Frecuencias**")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Histograma base
            ax.hist(
                df_temp_cont["_temp_continua"], 
                bins=cortes, 
                color='#10B981', 
                edgecolor='white', 
                alpha=0.55, 
                label='Histograma (fi)'
            )
            
            # Polígono uniendo las marcas de clase (Xi)
            ax.plot(
                tabla_agrupada['Xi (Marca de Clase)'], 
                tabla_agrupada['fi'], 
                color='#EF4444', 
                marker='D', 
                linewidth=2.5, 
                markersize=6,
                label='Polígono de Frecuencias'
            )
            
            ax.set_title(f"Distribución Continua de {var_continua.replace('_', ' ')}", pad=15)
            ax.set_xticks(cortes)
            ax.set_xlabel(f"Límites e Intervalos de Clase")
            ax.set_ylabel("Frecuencia Absoluta (fi)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_g2:
            st.markdown("**Ojiva (Frecuencia Acumulada)**")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Gráfica de Ojiva acumulativa
            ax.plot(
                tabla_agrupada['Xi (Marca de Clase)'], 
                tabla_agrupada['Fi'], 
                color='#3B82F6', 
                marker='s', 
                linewidth=2.5, 
                markersize=6,
                label='Curva de Ojiva (Fi)'
            )
            
            # Relleno del área bajo la curva
            ax.fill_between(
                tabla_agrupada['Xi (Marca de Clase)'], 
                tabla_agrupada['Fi'], 
                color='#3B82F6', 
                alpha=0.15
            )
            
            ax.set_title("Ojiva Acumulativa de Frecuencias", pad=15)
            ax.set_xticks(cortes)
            ax.set_xlabel("Intervalos de Clase")
            ax.set_ylabel("Frecuencia Absoluta Acumulada (Fi)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# FOOTER / PIE DE PÁGINA
# ==============================================================================
st.markdown("---")
st.caption("Plataforma de Procesamiento Estadístico Descriptivo. Diseñada bajo estándares de analítica avanzada.")