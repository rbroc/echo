import re

r_pre = {'。': '.',
         '′': "'",
         '': '',
         '~': '',
         '、': ',',
         '=': "",
         '\\\\': '', # double-check
         'M & M': 'M&M',
         "' great outdoors '": '"great outdoors"', 
         '" Fried Rice , Hangzhou Style "': '"Fried Rice, Hangzhou Style"',
         '" Fruit Salad "': '"Fruit Salad"',
         '" hello "': '"hello"',
         ' " Please finding': ': Please finding',
         '" Money makes the mare go . "': '"Money makes the mare go."',  
         '" Love , Amy Card "': 'Love , Amy Card',
         '" When I fastened the cord and walked to the platform , I was extremely nervous . When I looked down , I was nervous even more . If the cord had broken , I would be knocked to bakemeat . However , I knew my worries were unwanted . I was afraid if I retreated others might say something about me . I had to harden my heart and impose my head downward . I only felt wind wiring at my ears . My body rapidly descended . I thought that after only seven or eight seconds I was rebounded to upper air by the cord . In fact it was 30 seconds . Maybe because of my nervousness , I only felt seven or eight seconds . When rebounding to the high altitude , I felt a sudden sense of relief . I did not feel nervous very much , but very comfortable . "':
         '"When I fastened the cord and walked to the platform , I was extremely nervous . When I looked down , I was nervous even more . If the cord had broken , I would be knocked to bakemeat . However , I knew my worries were unwanted . I was afraid if I retreated others might say something about me . I had to harden my heart and impose my head downward . I only felt wind wiring at my ears . My body rapidly descended . I thought that after only seven or eight seconds I was rebounded to upper air by the cord . In fact it was 30 seconds . Maybe because of my nervousness , I only felt seven or eight seconds . When rebounding to the high altitude , I felt a sudden sense of relief . I did not feel nervous very much , but very comfortable."',
         '" card catalog "': '"card catalog"',  
         '" Happy Birthday "': '"Happy Birthday"',
         '" Good Morning to you "': '"Good Morning to you"',   
         '" Thanks "': '"Thanks"',
         '" Coal Hill "': '"Coal Hill"',
         '" cheese "' : '"cheese"',
         '" Tequila Sauta "' : '"Tequila Sauta"',
         '" If we get thirsty , we\'ll have something to drink , "': '"If we get thirsty, we\'ll have something to drink"',
         '" If we get lost , we\'ll be able to find our way . "': '"If we get lost , we\'ll be able to find our way."',
         '" Why the door ? "': '"Why the door?"',
         '" Well , "': 'Well, "',
         '“ If it gets hot , we can open the window . "': '"If it gets hot , we can open the window."',
         '" impossible is nothing "': '"impossible is nothing"',
         '" suffer "': '"suffer"',
         '" dim sum , "': '"dim sum"',  
         '" new "':'"new"',  
         '" Is there anything I can do ? "': '"Is there anything I can do ?"',
         '" This is Abby\'s voicemail . I will call you later , so leave me your name and number "': '"This is Abby\'s voicemail . I will call you later , so leave me your name and number"',
         '" This is Abby and I am really happy you called ! I promise I will give you a ring as soon as I can , so please leave me your name and number . Talk to you soon ! "': '"This is Abby and I am really happy you called ! I promise I will give you a ring as soon as I can , so please leave me your name and number . Talk to you soon !"',  
         '" Hi , you have reached Abby . I am unable to answer your call right now , but if you leave me your name and phone number , I will get back to you as soon as possible . Thanks "' : '"Hi , you have reached Abby . I am unable to answer your call right now , but if you leave me your name and phone number , I will get back to you as soon as possible . Thanks"', 
         'I " m': 'I am',  
         '" milk of human kindness "': 'milk of human kindness',
         '" Healthy eating is not about strict nutrition philosophies , staying unrealistically thin , or depriving yourself of foods you love . "': '"Healthy eating is not about strict nutrition philosophies , staying unrealistically thin , or depriving yourself of foods you love ."',
         '" strict nutrition philosophies "': '"strict nutrition philosophies"',
         '" goodbye "': '"goodbye"',
         '" for guests only "' : '"for guests only"',
         '" chi "': '"chi"',
         'Pauline is typing letters . She can\'t speak to you now ! "': '"Pauline is typing letters . She can\'t speak to you now!"',  
         '" yard sale "': '"yard sale"',
         '" Do you want me to help you ? "': '"Do you want me to help you?"',
         '" Chinese-Style Divorce "': '"Chinese-Style Divorce"',
         '" Four and One "': '"Four and One"',  
         '" Tie a Yellow Ribbon on the Old Oak Tree "': '"Tie a Yellow Ribbon on the Old Oak Tree"'
}

r_post = {'’': "'",
          '‘': "'",
          '”': '"',
          '“': '"'}
          

ws_after = ['!', ',', ';', '%', ':', '’', '”', '%']
ws_before = ['$', '£', '¥', '‘', '“']
ws_none = ['/', '&', '°']
ws_after_escape = ['.', ')', '?']
ws_before_escape = ['(']
ws_none_escape = ['*', '+']


def standardize_punctuation(t, lower=True):
    for k,v in r_pre.items():
        t = re.sub(k,v,t)
    for c in ws_after:
        t = re.sub(rf' {c} ', f'{c} ', t)
    for c in ws_before:
        t = re.sub(rf' {c} ', f' {c}', t)
    for c in ws_none:
        t = re.sub(rf' {c} ', f'{c}', t)
    for c in ws_after_escape:
        t = re.sub(rf' \{c} ', f'{c} ', t)
    for c in ws_before_escape:
        t = re.sub(rf' \{c} ', f' {c}', t)
    for c in ws_none_escape:
        t = re.sub(rf' \{c} ', f'{c}', t)
    for k,v in r_post.items():
        t = re.sub(k,v,t)
    if lower is True:
        t = t.lower()
    t = re.sub(' +', ' ', t)
    t = t.strip()
    return t

