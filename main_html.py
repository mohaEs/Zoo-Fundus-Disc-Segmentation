
import Collected_Libraries.HTML as HTML
import os

path_images='./tmp_images'
path_res_DENet='./results_DENet'
path_res_MNet='./results_MNet'
path_res_AttnNet='./results_AttnUnet'
path_res_AdaTh='./results_AdaptTh'
# open an HTML file to show output in a browser
HTMLFILE = 'index.html'
f = open(HTMLFILE, 'w')
t = HTML.Table(header_row=['filename','input', 'DENet', 'MNet','AttnNet','AdaptiveThr'])

#print(imgespath)
for root, dirs, files in os.walk(path_images):
    for filename in files:
        # print('===> filename', filename)
        t.rows.append([filename,
        str("<img src=" + os.path.join(path_images,filename) +" width=256 height=256 />"),
        "<img src=" + os.path.join(path_res_DENet,filename[:-3]+'png') +" width=256 height=256 />",
        "<img src=" + os.path.join(path_res_MNet,filename[:-3]+'png') +" width=256 height=256 />",
        "<img src=" + os.path.join(path_res_AttnNet,filename) +" width=256 height=256 />",
        "<img src=" + os.path.join(path_res_AdaTh,filename[:-3]+'png') +" width=256 height=256 />"
        ])

htmlcode = str(t)
f.write(htmlcode)
f.write('<p>')