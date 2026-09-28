"""Official meeting-ending dates, transcribed from Federal Reserve annual calendars.
The first 48 dates anchor Figure 5; 2000-2016 dates come from the frozen panel.
These are dates, not manually coded tone labels.
"""
OFFICIAL={
1994:['02-04','03-22','05-17','07-06','08-16','09-27','11-15','12-20'],
1995:['02-01','03-28','05-23','07-06','08-22','09-26','11-15','12-19'],
1996:['01-31','03-26','05-21','07-03','08-20','09-24','11-13','12-17'],
1997:['02-05','03-25','05-20','07-02','08-19','09-30','11-12','12-16'],
1998:['02-04','03-31','05-19','07-01','08-18','09-29','11-17','12-22'],
1999:['02-03','03-30','05-18','06-30','08-24','10-05','11-16','12-21']}
SOURCES={year:f'https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm' for year in OFFICIAL}
