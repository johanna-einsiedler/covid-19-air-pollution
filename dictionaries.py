# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:30:33 2020

@author: hicom
"""


starts = {'che': '03/16/2020',
          'beijing': '01/23/2020',
          'wuhan': '01/23/2020',
          'at': '03/20/2020'}

ends = {'che': '04/26/2020',
        'beijing': '04/08/2020',
        'wuhan': '04/08/2020',
        'at': '04/26/2020'}


ex_vars_groups = {'ws': 'WS',
                  'h': 'RH',
                  'p': 'P',
                  'month': 'M',
                  'wx': 'WD_X',
                  'wy': 'WD_Y',
                  't': 'T',
                  'julian': 'JD',
                  'h_lag1': 'RH',
                  't_lag1': 'T',
                  'wx_lag1': 'WD_X',
                  'wy_lag1': 'WD_Y',
                  'ws_lag1': 'WS',
                  'p_lag1': 'P',
                  'h_lag1': 'RH',
                  'h_lag2': 'RH',
                  't_lag2': 'T',
                  'wx_lag2': 'WD_X',
                  'wy_lag2': 'WD_Y',
                  'ws_lag2': 'WS',
                  'p_lag2': 'P',
                  'h_lag2': 'RH',
                  'h_lag3': 'RH',
                  't_lag3': 'T',
                  'wx_lag3': 'WD_X',
                  'wy_lag3': 'WD_Y',
                  'ws_lag3': 'WS',
                  'p_lag3': 'P',
                  'h_lag3': 'RH',
                  'lagws_1week': 'WS',
                  'lagws_2weeks': 'WS',
                 'dew': 'DEW',
                  'dew_lag1': 'DEW',
                  'dew_lag2': 'DEW',
                  'dew_lag3': 'DEW',
                  'lagpca_1week': 'PCA',
                  'lagpca_2weeks': 'PCA',
                  'lagpca_4weeks': 'PCA',
                  'lagws_1week_max': 'WS',
                  'lagws_2weeks_max': 'WS',
                  'lagws_4weeks': 'WS',
                  'lagws_4weeks_max': 'WS',
                  'pca': 'PCA',
                  'lagpca_8weeks': 'PCA',
                  'lagpca_12weeks': 'PCA'
                  
                 }


che_classes = {'Zuerich_Schimmelstrasse': 'low traffic', 'Zuerich_Stampfenbachstrasse': 'low traffic',
          'StGallen_Stuelegg': 'no traffic', 'StGallen_Blumenbergplatz': 'low traffic', 'Opfikon_Balsberg': 'high traffic'}

beijing_classes = {'aotizhongxin': 'urban', 
           'badaling': 'suburban',
           'beibuxinqu': 'urban',
           'changping': 'rural',
           'daxing': 'rural',
           'dingling':'suburban', 
            'donggaocun': 'suburban',
           'dongsi':'urban',
           'dongsihuan': 'road',
           'fangshan': 'rural',
           'fengtaihuayuan':'urban',
           'guanyuan':'urban', 
           'gucheng':'urban', 
           'huairou':'rural', 
           'liulihe': 'suburban',
           'mentougou': 'rural',
           'miyun':'rural',
           'miyunshuiku': 'suburban',
           'nansanhuan': 'road',
           'nongzhanguan':'urban',
           'pinggu':'rural',
           'qianmen': 'road',
           'shunyi':'rural',
           'tiantan': 'urban',
           'tongzhou': 'rural',
           'wanliu':'urban',
           'wanshouxigong':'urban',
           'xizhimenbei': 'road',
           'yanqing':'rural',
           'yizhuang': 'rural',
           'yongdingmennei': 'road',
           'yongledian': 'suburban',
           'yufa': 'suburban',
           'yungang':'urban'}


wuhan_classes = {'1325A': '1325A',
                 '1326A': '1326A',
                 '1327A': '1327A', 
                 '1328A': '1328A',
                 '1329A': '1329A', 
                 '1330A': '1330A', 
                 '1331A': '1331A', 
                 '1333A': '1333A', 
                 '1334A': '1334A'}


at_classes = {'amstetten': 'suburban_residential',
              'annaberg': 'rural',
              'biedermannsdorf': 'residential',
              'dunkelsteinerwald': 'rural',
              'forsthof': 'rural',
              'heidenreichstein': 'rural',
              'himberg': 'suburban_residential',
              'irnfritz': 'rural',
              'kematenybbs': 'rural',
              'klosterneuburg': 'suburban_residential',
              'klosterneuburg-verkehr': 'urban',
              'kollmitzberg': 'rural',
              'krems': 'residential',
              'mistelbach': 'rural',
              'moedling': 'residential',
              'neusiedl': 'unknown',
              'payerbach': 'rural',
              'poechlarn': 'residential',
              'purkersdorf': 'residential',
              'schwechat': 'urban',
              'st.valentin-a1': 'urban',
              'stixneusiedl': 'rural',
              'stockerau': 'residential',
              'stpoelten': 'urban',
              'streithofen': 'suburban_residential',
              'traismauer': 'suburban_residential',
              'tulln': 'suburban_residential',
              'voesendorf': 'residential',
              'wienerneudorf': 'residential',
              'wienerneustadt': 'suburban_residential',
              'wiesmath': 'rural',
              'wolkersdorf': 'rural',
              'ziersdorf': 'rural',
              'zwentendorf': 'suburban_residential'}


loc_classes = {'che': che_classes,
               'beijing': beijing_classes,
               'wuhan': wuhan_classes,
               'at': at_classes}

