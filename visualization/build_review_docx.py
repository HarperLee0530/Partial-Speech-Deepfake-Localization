from pathlib import Path
import argparse
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from visualization_paths import analysis_directory, project_path, output_path
parser=argparse.ArgumentParser(description='Build the Word review document from existing diagnostic figures.')
parser.add_argument('--analysis-dir',help='Diagnostics directory, relative to project root; default: latest completed diagnostics')
parser.add_argument('--output',help='DOCX destination, relative to project root')
args=parser.parse_args()
ROOT=analysis_directory(args.analysis_dir)/'导师审阅'
FIG=ROOT/'figures'
OUT=output_path(args.output) if args.output else ROOT/'SAL基线诊断与区间修正研究方案.docx'
OUT.parent.mkdir(parents=True,exist_ok=True)
doc=Document()
sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.8);sec.bottom_margin=Cm(1.8)
sec.left_margin=Cm(2);sec.right_margin=Cm(2)
sec.footer_distance=Cm(.8)

def fontstyle(style,latin,east,size,bold=False):
    style.font.name=latin;style.font.size=Pt(size);style.font.bold=bold;style.font.color.rgb=RGBColor(0,0,0)
    rpr=style.element.get_or_add_rPr()
    fonts=rpr.find(qn('w:rFonts'))
    if fonts is None:fonts=OxmlElement('w:rFonts');rpr.insert(0,fonts)
    for key,val in [('ascii',latin),('hAnsi',latin),('eastAsia',east)]:fonts.set(qn('w:'+key),val)
    for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:fonts.attrib.pop(qn('w:'+key),None)

fontstyle(doc.styles['Normal'],'Times New Roman','宋体',11)
pf=doc.styles['Normal'].paragraph_format
pf.line_spacing=Pt(16);pf.space_after=Pt(6);pf.first_line_indent=Pt(22)
pf.widow_control=True
fontstyle(doc.styles['Title'],'Times New Roman','微软雅黑',17,True)
doc.styles['Title'].paragraph_format.space_after=Pt(8)
doc.styles['Title'].paragraph_format.first_line_indent=Pt(0)
fontstyle(doc.styles['Heading 1'],'Times New Roman','微软雅黑',13,True)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(8)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(6)
doc.styles['Heading 1'].paragraph_format.first_line_indent=Pt(0)
fontstyle(doc.styles['Heading 2'],'Times New Roman','微软雅黑',11.5,True)
doc.styles['Heading 2'].paragraph_format.space_before=Pt(6)
doc.styles['Heading 2'].paragraph_format.space_after=Pt(6)
doc.styles['Heading 2'].paragraph_format.first_line_indent=Pt(0)
fontstyle(doc.styles['Caption'],'Times New Roman','宋体',9.5)
doc.styles['Caption'].paragraph_format.first_line_indent=Pt(0)
doc.styles['Caption'].paragraph_format.line_spacing=Pt(12)
doc.styles['Caption'].paragraph_format.space_after=Pt(5)

def p(text,style=None):
    paragraph=doc.add_paragraph(text,style)
    if not style:paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT
    return paragraph

def heading(text,level=1):return doc.add_heading(text,level)

def figure(num,name,title,caption):
    para=doc.add_paragraph()
    para.paragraph_format.first_line_indent=Pt(0)
    para.paragraph_format.space_after=Pt(3)
    para.paragraph_format.line_spacing=1
    para.paragraph_format.keep_with_next=True
    para.alignment=WD_ALIGN_PARAGRAPH.CENTER
    shape=para.add_run().add_picture(str(FIG/(name+'.png')),width=Cm(16.5 if num==1 else 17))
    shape._inline.docPr.set('descr',title+'。'+caption)
    cap=p(f'图{num}  {title}','Caption')
    cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].bold=True
    cap.paragraph_format.keep_with_next=True
    note=p(caption,'Caption')
    note.alignment=WD_ALIGN_PARAGRAPH.LEFT

footer=sec.footer.paragraphs[0]
footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.paragraph_format.first_line_indent=Pt(0)
run=footer.add_run();run.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
run._r.addnext(field)
doc.core_properties.title='SAL基线诊断与区间修正研究方案'
doc.core_properties.subject='基线错误证据 设计动机 拟议方法与待验证实验'
doc.core_properties.author='项目组'

doc.add_paragraph('SAL基线诊断与区间修正研究方案','Title')
heading('1 基线诊断与问题发现')
p('我们首先对已保存的 SAL 预测进行事后错误分析，分别考察 PartialSpoof 测试集及 LlamaPartialSpoof crossfade 评估集。分析在各自原生评价网格上进行，使用固定的二分类决策规则，并将局部伪造录音与全真、全伪造录音分开统计。以下事件统计均针对局部伪造录音。')
p('如图1a–g所示，SAL 的错误不仅包括整段漏检，还包括误报、区间收缩或外扩、区间偏移、预测碎片化及错误合并。这些案例说明，剩余错误涉及不同的区间结构问题，单一的边界扩张或区间合并策略可能无法同时处理这些情况。')
figure(1,'图1_PS时间轴诊断','PartialSpoof上SAL预测的时间轴诊断',
 '青蓝横条为真实伪造区间，紫色横条为预测区间，深蓝曲线为未经校准的伪造类分数，虚线为固定阈值0.5。a漏检；b误报；c收缩；d外扩；e偏移；f碎片化；g错误合并；h正确对照。真值与预测均采用160 ms网格。案例展示错误形式，不代表发生频率。')

doc.add_page_break()
heading('PartialSpoof的时长与边界表现',2)
p('如图2a和图2b所示，PartialSpoof 上最短网格事件数量较多，且完全漏检率明显高于其他时长组。在160 ms评价网格上，时长不超过160 ms的事件占全部真实伪造事件的44.49%，却贡献了89.16%的完全漏检，其组内完全漏检率为3.67%。相比之下，160–320 ms组的完全漏检率为0.74%。事件占比和漏检贡献比例由图2对应的分组计数计算得到。')
figure(2,'图2_PS时长与边界统计','PartialSpoof上按事件时长分组的错误分析',
 'a为真实事件数量；b分别报告完全无重叠的漏检率和未获得一对一tIoU≥0.5匹配的事件比例，误差线为400次按录音重采样的95%区间；c、d为匹配成功事件的起点、终点偏差，各列为百分比，色深表示占比。统计包含60,988条局部伪造录音和194,003个真实事件。同源变体可能相关，区间不视为严格源独立的显著性证据。')
p('完全漏检指真实事件与所有预测区间均无正时长重叠。tIoU为两区间交集时长与并集时长之比；一对一匹配先最大化tIoU≥0.5的合格匹配数量，再最大化匹配tIoU之和。未匹配比例包含完全漏检，两者不能相加。')
p('如图2c和图2d所示，已匹配事件的大部分起止边界与评价网格标注一致。这里的一致仅指与160 ms网格一致，不代表原始音频上的精确编辑边界已被完全恢复；较粗网格也可能合并相邻事件或抹去短间隔。最短组主要是单网格事件，不能直接等同于精细标注下的全部短伪造片段。')

doc.add_page_break()
heading('LlamaPartialSpoof的区间错误结构',2)
p('如图3所示，在 LlamaPartialSpoof crossfade 的20 ms评价网格上，完全漏检率仅为0.14%，但未获得一对一tIoU≥0.5匹配的真实事件比例达到70.49%，对应事件召回率为29.51%。同时，29.29%的真实事件出现预测碎片化，39.50%的预测区间与真实伪造区间没有重叠。这表明，预测与伪造事件存在少量交集，并不足以说明模型已经准确定位该事件。')
figure(3,'图3_LPS错误类型统计','LlamaPartialSpoof crossfade上的区间错误结构',
 '统计包含32,194条局部伪造录音、113,953个真实事件及254,649个预测区间。完全漏检、未匹配及碎片化以真实事件为分母；错误合并和无重叠误报以预测区间为分母。各类别可能重叠，不能相加。事件匹配阈值为tIoU≥0.5，决策规则固定。')
heading('2 设计动机')
p('上述结果说明，有必要同时考虑事件是否被检出，以及预测区间的完整性、分离情况和误报代价。图1f和图1g分别展示了需要合并预测碎片、以及需要拆分错误连接区间的情况，说明不同局部错误可能需要方向相反的修改。因此，我们研究在初始声学预测之后增加区间修正机制，依据局部证据评估不同修改对事件级错误的影响。')
p('与此同时，如图2c和图2d所示，PartialSpoof 中已匹配事件的大部分起止边界与评价网格标注一致；图1h也展示了应保留原预测的正确案例。因此，修正机制应具备保留原预测的能力，避免不必要的修改破坏已有正确结果。')
p('当前诊断支持对不同错误采取有条件的区间决策，但尚未确定这些错误的因果机制，也未验证局部复检、语义辅助或收益回归的实际效果。两套结果采用不同数据和评价网格，不能将其差异直接归因于某一个因素。')

doc.add_page_break()
heading('3 拟议方法')
p('针对图1展示的不同预测结果，我们将区间处理建模为六种候选决策：保留、插入、删除、拆分、合并和边界调整。其中，保留不改变原预测，其余五类操作对区间进行修改。各决策与诊断现象的对应关系如下。')
table=doc.add_table(rows=1,cols=3)
table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
widths=[2.5,3.1,11.4]
for c,w,text in zip(table.rows[0].cells,widths,['候选决策','对应子图','目标现象']):c.width=Cm(w);c.text=text
mapping=[('保留','图1h','保留已经正确的预测，避免有害修改'),('插入','图1a','恢复初次未形成预测区间的伪造事件'),
 ('删除','图1b','移除与真实伪造区间无重叠的误报'),('拆分','图1g','分开被错误连接的多个真实事件'),
 ('合并','图1f','连接同一真实事件内部的预测碎片'),('边界调整','图1c–e','修正区间收缩、外扩及起止位置偏移')]
for values in mapping:
    for c,w,text in zip(table.add_row().cells,widths,values):c.width=Cm(w);c.text=text
pr=table._tbl.tblPr
borders=OxmlElement('w:tblBorders')
for side in ['top','left','bottom','right','insideH','insideV']:
    edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
pr.append(borders)
for i,row in enumerate(table.rows):
    cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
    for j,cell in enumerate(row.cells):
        cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        tcpr=cell._tc.get_or_add_tcPr()
        margins=OxmlElement('w:tcMar')
        for side,val in [('top','100'),('bottom','100'),('left','100'),('right','100')]:
            el=OxmlElement('w:'+side);el.set(qn('w:w'),val);el.set(qn('w:type'),'dxa');margins.append(el)
        tcpr.append(margins)
        if i==0:
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'A7D7E8');tcpr.append(shade)
        for par in cell.paragraphs:
            par.paragraph_format.first_line_indent=Pt(0);par.paragraph_format.space_after=Pt(0)
            par.paragraph_format.line_spacing=Pt(14)
            par.alignment=WD_ALIGN_PARAGRAPH.CENTER if j<2 else WD_ALIGN_PARAGRAPH.LEFT
            for r in par.runs:r.font.size=Pt(10);r.bold=i==0
hdr=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(hdr)
after=p('修正器根据局部声学信息、初始分数及区间结构，预测候选操作带来的错误降低量，并仅接受预期收益足够高的修改。图1a还表明，至少存在未形成任何预测区间的漏检案例，因此候选生成不能仅限于初次已经检出的区间。我们将允许系统在初次未标出的区域提出插入候选。')
after.paragraph_format.space_before=Pt(8)
p('对于初始证据不足的区域，我们进一步研究改变上下文的局部声学复检，并通过预算约束控制额外计算。词汇与语义类别统计作为复检位置选择的辅助输入，其有效性需要通过与纯声学选择策略的对照实验验证。')
heading('4 待验证假设与实验安排')
p('我们将分别检验：改变上下文的复检是否提供可利用的补充证据；收益监督是否优于简单规则及操作分类；语义信息是否在相同候选池和计算预算下改善复检选择。')
p('鉴于图3中完全漏检率较低与区间匹配失败率较高同时存在，我们不会仅以覆盖率提升作为复检有效的依据。实验将同时报告事件召回、区间匹配质量、误报、有害修改和计算开销，以区分错误修复带来的收益与额外误报造成的代价。')
p('现有图表为已有测试结果的事后诊断，用于形成研究假设。后续方法选择与参数调整将在开发数据上完成；对已用于形成方案的测试集结果，将如实说明其用途，并尽可能使用额外未参与决策的数据验证泛化。')
# Remove inherited title/paragraph borders from the bundled default template.
for root in [doc.styles.element,doc._element]:
    for border in list(root.iter(qn('w:pBdr'))):border.getparent().remove(border)
doc.save(OUT)
print(str(OUT))
