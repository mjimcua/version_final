-- 2.1 · la curva de maduración de cada estado con fechas y su examen por antelación
SELECT estado, meses_antes, parte_que_entra, parte_visible, p_entrar, tope, se_usa, error_curva_pp, error_sin_curva_pp, cohortes
FROM sff_curva_maduracion
ORDER BY estado, meses_antes;
