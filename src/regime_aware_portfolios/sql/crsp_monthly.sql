-- Monthly stock file of CRSP, common stocks (share codes 10 and 11), with the
-- exchange and industry codes in force at each date and any delisting return of
-- the month. The query follows Tidy Finance with Python (Scheuch, Voigt, Weiss and
-- Frey), "WRDS, CRSP, and Compustat".
SELECT msf.permno, msf.date, date_trunc('month', msf.date)::date as month, msf.ret, msf.shrout, msf.vol, msf.prc, msf.altprc, msf.bid, msf.ask, msenames.exchcd, msenames.siccd, msedelist.dlret, msedelist.dlstcd
FROM crsp.msf AS msf
LEFT JOIN crsp.msenames as msenames ON msf.permno = msenames.permno AND msenames.namedt <= msf.date AND msf.date <= msenames.nameendt
LEFT JOIN crsp.msedelist as msedelist
ON msf.permno = msedelist.permno AND date_trunc('month', msf.date)::date = date_trunc('month', msedelist.dlstdt)::date
WHERE msf.date BETWEEN %(start)s AND %(end)s AND msenames.shrcd IN (10, 11)
