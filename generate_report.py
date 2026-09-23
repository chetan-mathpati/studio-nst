from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "Studio-NST-Technical-Report.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=18 * mm,
    leftMargin=18 * mm,
    topMargin=18 * mm,
    bottomMargin=18 * mm,
)

styles = getSampleStyleSheet()

title = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontSize=25,
    leading=30,
    alignment=TA_CENTER,
    spaceAfter=8,
)

subtitle = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    leading=16,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"),
    spaceAfter=24,
)

h1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=17,
    leading=21,
    spaceBefore=14,
    spaceAfter=9,
)

h2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=12.5,
    leading=16,
    spaceBefore=9,
    spaceAfter=5,
)

body = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=14,
    spaceAfter=7,
)

small = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11,
    textColor=colors.HexColor("#555555"),
)

bullet = ParagraphStyle(
    "Bullet",
    parent=body,
    leftIndent=12,
    firstLineIndent=-7,
)

code = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=7.5,
    leading=10,
    backColor=colors.HexColor("#F3F3F3"),
    borderPadding=6,
    spaceBefore=4,
    spaceAfter=8,
)

story = []

story.append(Spacer(1, 25 * mm))
story.append(Paragraph("Studio NST", title))
story.append(Paragraph("Neural Style Transfer & Visual Experimentation", subtitle))
story.append(Paragraph(
    "Technical Report",
    ParagraphStyle(
        "ReportLabel",
        parent=styles["Heading2"],
        alignment=TA_CENTER,
        fontSize=13,
        spaceAfter=30,
    ),
))
story.append(Paragraph(
    "A research-oriented implementation and experimental evaluation of three neural style transfer approaches.",
    ParagraphStyle(
        "CoverText",
        parent=body,
        alignment=TA_CENTER,
        fontSize=11,
        leading=17,
    ),
))
story.append(Spacer(1, 30 * mm))
story.append(Paragraph(
    "Experimental environment: NVIDIA GeForce GTX 1650 · 4 GB VRAM · CUDA 13.0 · PyTorch 2.14.0+cu130 · Python 3.14",
    small,
))
story.append(PageBreak())

story.append(Paragraph("1. Abstract", h1))
story.append(Paragraph(
    "Studio NST is a research-oriented implementation of neural style transfer that brings three distinct approaches into one reproducible experimental environment: optimization-based style transfer inspired by Gatys et al., feed-forward style transfer based on the Johnson et al. approach, and arbitrary style transfer using Adaptive Instance Normalization (AdaIN).",
    body,
))
story.append(Paragraph(
    "The project evaluates computational behavior through runtime, GPU memory, optimization losses, and visual inspection. Experiments were performed on a consumer NVIDIA GeForce GTX 1650 with 4 GB of GPU memory.",
    body,
))
story.append(Paragraph(
    "The results demonstrate the different computational characteristics of iterative optimization, feed-forward transformation, and feature-statistics-based arbitrary style transfer. The Johnson experiment is explicitly treated as a pipeline validation because its limited training setup did not produce a sufficiently trained model for a meaningful quality comparison.",
    body,
))

story.append(Paragraph("2. Research Motivation", h1))
story.append(Paragraph(
    "Neural style transfer demonstrates how deep visual representations can be used to separate image content from artistic style and recombine them into a new image.",
    body,
))
story.append(Paragraph(
    "Studio NST studies three approaches representing different computational strategies. Gatys performs direct optimization of the generated image. Johnson learns a transformation network that can perform feed-forward inference. AdaIN aligns feature statistics between content and style representations and uses a decoder to reconstruct the stylized image.",
    body,
))

story.append(Paragraph("3. Research Questions", h1))
questions = [
    "How does optimization-based style transfer behave as the number of optimization steps increases?",
    "How does input resolution affect runtime and GPU memory usage?",
    "What practical differences exist between iterative optimization and feed-forward transformation?",
    "How does AdaIN enable arbitrary style transfer?",
    "How does AdaIN style strength affect the interpolation between content and stylized representations?",
    "What are the practical constraints of neural style transfer on a 4 GB GPU?",
]
for item in questions:
    story.append(Paragraph("• " + item, bullet))

story.append(Paragraph("4. Methods", h1))

story.append(Paragraph("4.1 Gatys", h2))
story.append(Paragraph(
    "The Gatys implementation directly optimizes the generated image. A pretrained VGG19 network provides intermediate feature representations. Content similarity is represented through VGG features, while style is represented using Gram matrices computed from multiple feature layers.",
    body,
))
story.append(Paragraph(
    "The implementation exposes configurable optimization steps, content weight, style weight, and learning rate.",
    body,
))

story.append(Paragraph("4.2 Johnson", h2))
story.append(Paragraph(
    "The Johnson implementation uses a feed-forward transformation network containing convolutional feature extraction, downsampling layers, residual blocks, and upsampling layers. Training uses perceptual losses derived from VGG19 features.",
    body,
))
story.append(Paragraph(
    "The current experiment uses a limited 200-image CIFAR-10 subset to validate the training and inference pipeline. It is not treated as a production-quality training configuration.",
    body,
))

story.append(Paragraph("4.3 AdaIN", h2))
story.append(Paragraph(
    "Adaptive Instance Normalization aligns the channel-wise mean and standard deviation of content features with those of style features. The resulting representation is passed through a pretrained decoder.",
    body,
))
story.append(Paragraph(
    "This allows arbitrary style images to be used without training a separate transformation network for each style.",
    body,
))

story.append(PageBreak())

story.append(Paragraph("5. Experimental Setup", h1))
setup = [
    ["Component", "Configuration"],
    ["GPU", "NVIDIA GeForce GTX 1650"],
    ["GPU Memory", "4 GB"],
    ["CUDA Runtime", "13.0"],
    ["PyTorch", "2.14.0+cu130"],
    ["Python", "3.14"],
    ["Operating System", "Windows"],
]
table = Table(setup, colWidths=[55 * mm, 105 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("LEADING", (0, 0), (-1, -1), 11),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("PADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Runtime was measured using Python's high-resolution performance counter. CUDA synchronization was used where required around GPU operations. Peak GPU memory was measured using PyTorch CUDA memory statistics.",
    body,
))

story.append(Paragraph("6. Gatys Step Scaling", h1))
story.append(Paragraph(
    "The same content and style image pair was evaluated at a maximum size of 256 pixels using 100, 300, and 500 optimization steps.",
    body,
))
gatys_steps = [
    ["Steps", "Runtime", "Peak GPU Memory", "Content Loss", "Style Loss"],
    ["100", "5.0686 s", "264.57 MB", "10.795232", "7.127049e-06"],
    ["300", "13.5208 s", "263.63 MB", "9.615572", "5.248773e-06"],
    ["500", "22.5941 s", "263.63 MB", "9.292706", "4.966042e-06"],
]
table = Table(gatys_steps, colWidths=[25 * mm, 30 * mm, 36 * mm, 34 * mm, 34 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("LEADING", (0, 0), (-1, -1), 10),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 5),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Increasing optimization steps increased runtime while continuing to reduce the recorded optimization losses. The improvement in content loss became smaller as additional optimization steps were added.",
    body,
))

story.append(Paragraph("7. Gatys Resolution Scaling", h1))
story.append(Paragraph(
    "The same image pair was evaluated at 256, 384, and 512 pixel maximum resolutions using 300 optimization steps.",
    body,
))
gatys_resolution = [
    ["Resolution", "Runtime", "Peak GPU Memory", "Content Loss", "Style Loss"],
    ["256 px", "14.1449 s", "264.57 MB", "9.591365", "5.511013e-06"],
    ["384 px", "29.0578 s", "451.32 MB", "6.550163", "2.810368e-06"],
    ["512 px", "53.6524 s", "729.36 MB", "5.367336", "2.035976e-06"],
]
table = Table(gatys_resolution, colWidths=[25 * mm, 30 * mm, 36 * mm, 34 * mm, 34 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ("LEADING", (0, 0), (-1, -1), 10),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 5),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Increasing resolution substantially increased runtime and peak GPU memory in the recorded experiment. The recorded losses also decreased with resolution, but the losses alone do not establish that higher resolution produced better perceived image quality.",
    body,
))

story.append(PageBreak())

story.append(Paragraph("8. Johnson Pipeline Validation", h1))
story.append(Paragraph(
    "The Johnson experiment used a limited 200-image CIFAR-10 subset and 50 training steps. The complete training and inference pipeline executed successfully.",
    body,
))
johnson = [
    ["Measurement", "Initial", "Final"],
    ["Total Loss", "325.259888", "101.421913"],
    ["Content Loss", "9.104837", "8.196975"],
    ["Style Loss", "3.161550e-03", "9.322494e-04"],
]
table = Table(johnson, colWidths=[65 * mm, 47 * mm, 47 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("PADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph("Training runtime: 6.15 seconds.", body))
story.append(Paragraph(
    "Although the pipeline completed successfully, the generated image did not provide satisfactory style transfer quality. The experiment is therefore treated as pipeline validation rather than evidence of a sufficiently trained feed-forward style transfer model.",
    body,
))

story.append(Paragraph("9. AdaIN Inference", h1))
adain = [
    ["Measurement", "Result"],
    ["Resolution", "512 px maximum"],
    ["Alpha", "1.0"],
    ["Runtime", "0.8758 s"],
    ["Peak GPU Memory", "147.68 MB"],
    ["Device", "NVIDIA GeForce GTX 1650"],
]
table = Table(adain, colWidths=[65 * mm, 94 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("PADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The generated result preserved the main structure of the content image while transferring visible characteristics of the style image.",
    body,
))

story.append(Paragraph("10. AdaIN Style Strength Experiment", h1))
story.append(Paragraph(
    "The same content and style image pair was evaluated at a maximum size of 512 pixels using five AdaIN alpha values. Alpha controls the strength of the style-transfer interpolation.",
    body,
))
adain_alpha = [
    ["Alpha", "Runtime", "Peak GPU Memory"],
    ["0.00", "0.4021 s", "147.68 MB"],
    ["0.25", "0.2863 s", "147.68 MB"],
    ["0.50", "0.2627 s", "147.68 MB"],
    ["0.75", "0.2944 s", "147.68 MB"],
    ["1.00", "0.2566 s", "147.68 MB"],
]
table = Table(adain_alpha, colWidths=[45 * mm, 55 * mm, 60 * mm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9ECEF")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Peak GPU memory remained constant at 147.68 MB across the tested alpha values. Runtime varied between 0.2566 and 0.4021 seconds across the individual runs. Because each alpha value was measured once, these runtime differences should not be interpreted as evidence that alpha itself changes inference speed.",
    body,
))

story.append(Paragraph("11. Evaluation", h1))
story.append(Paragraph(
    "Studio NST uses runtime, peak GPU memory, optimization losses, and visual inspection as evaluation dimensions.",
    body,
))
story.append(Paragraph(
    "Optimization losses describe the mathematical objective used during optimization and are not direct perceptual quality scores. Visual inspection is therefore reported separately from numerical measurements.",
    body,
))

story.append(Paragraph("12. Limitations", h1))
limitations = [
    "The primary visual evaluation uses one content image and one style image.",
    "The Johnson experiment uses only 200 CIFAR-10 images for pipeline validation.",
    "The Johnson model was not trained sufficiently for a meaningful quality comparison.",
    "Optimization losses are not direct measures of human-perceived image quality.",
    "Experiments were performed on a single GTX 1650 with 4 GB of GPU memory.",
    "Runtime and memory measurements can vary across hardware and system conditions.",
    "The AdaIN alpha experiment uses one content/style pair and one run per alpha value.",
    "Video style transfer and temporal consistency were not evaluated.",
]
for item in limitations:
    story.append(Paragraph("• " + item, bullet))

story.append(PageBreak())

story.append(Paragraph("13. Reproducibility", h1))
commands = [
    "python scripts/run_gatys.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --steps 500 --output outputs/gatys/",
    "python scripts/benchmark.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --max-size 256",
    "python scripts/benchmark_resolution.py --content assets/content/content1.jpg --style assets/styles/style1.jpg",
    "python scripts/run_adain.py --content assets/content/content1.jpg --style assets/styles/style1.jpg --output outputs/adain/stylized.png --max-size 512",
]
for command in commands:
    story.append(Paragraph(command, code))

story.append(Paragraph("14. Architecture", h1))
story.append(Paragraph(
    "The repository separates the three methods into independent implementations while sharing image processing, model utilities, experiment scripts, output handling, and measurement infrastructure.",
    body,
))
story.append(Paragraph(
    "The detailed system architecture is documented in docs/architecture/system-design.md.",
    body,
))

story.append(Paragraph("15. Future Work", h1))
future = [
    "Train the Johnson model on a larger and more appropriate image dataset.",
    "Evaluate a broader collection of content and style images.",
    "Introduce perceptual evaluation methods and human evaluation.",
    "Evaluate AdaIN alpha values across multiple content and style image pairs.",
    "Benchmark CPU and additional GPU hardware.",
    "Evaluate batch inference and larger resolutions.",
    "Investigate video style transfer and temporal consistency.",
    "Explore model compression and reduced-precision inference.",
    "Build an interactive style transfer interface.",
    "Automate the complete experiment suite and standardized reporting.",
]
for item in future:
    story.append(Paragraph("• " + item, bullet))

story.append(Paragraph("16. References", h1))
references = [
    "Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. https://arxiv.org/abs/1508.06576",
    "Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual Losses for Real-Time Style Transfer and Super-Resolution. https://arxiv.org/abs/1603.08155",
    "Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. https://arxiv.org/abs/1703.06868",
    "Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/pdf/1409.1556",
]
for reference in references:
    story.append(Paragraph(reference, body))

story.append(Spacer(1, 12))
story.append(Paragraph(
    "Studio NST is an experimental research implementation. Results are specific to the documented configurations and hardware environment.",
    small,
))


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Studio NST · {doc.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

print(f"PDF generated: {OUTPUT}")