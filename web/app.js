const $=s=>document.querySelector(s)
let method='adain'
let lastOriginal=null
let lastResult=null
const contents=[
  {name:'Your skyline',kind:'Local',url:'/assets/content/content1.jpg'},
  {name:'New York',kind:'Architecture',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/newyork.jpg'},
  {name:'Golden Gate',kind:'Landscape',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/golden_gate.jpg'},
  {name:'Flowers',kind:'Nature',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/flowers.jpg'},
  {name:'Chicago',kind:'City',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/chicago.jpg'},
  {name:'Sailboat',kind:'Landscape',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/content_data/sailboat.jpg'}
]
const styles=[
  {name:'Your landscape',kind:'Local',url:'/assets/styles/style1.jpg'},
  {name:'Sketch',kind:'Linework',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/sketch.png'},
  {name:'Brushstrokes',kind:'Painterly',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/brushstrokes.jpg'},
  {name:'Mondrian',kind:'Geometric',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/mondrian.jpg'},
  {name:'Picasso',kind:'Cubist',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/picasso_seated_nude_hr.jpg'},
  {name:'Matisse',kind:'Figurative',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/woman_with_hat_matisse.jpg'},
  {name:'La Muse',kind:'Portrait',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/la_muse.jpg'},
  {name:'Scene de Rue',kind:'Painterly',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/scene_de_rue.jpg'},
  {name:'Goeritz',kind:'Abstract',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/goeritz.jpg'},
  {name:'Flower of Life',kind:'Pattern',url:'https://raw.githubusercontent.com/shradha-khapra/ai-nst-project/main/NST_Code/style_data/flower_of_life.jpg'}
]
function renderLibrary(list,target,type){
  const root=$(target)
  root.innerHTML=list.map((x,i)=>`<button class="library-card" data-type="${type}" data-index="${i}"><img src="${x.url}" alt="${x.name}"><span class="card-number">${String(i+1).padStart(2,'0')}</span><span class="card-copy"><strong>${x.name}</strong><small>${x.kind}</small></span></button>`).join('')
}
renderLibrary(contents,'#contentLibrary','content')
renderLibrary(styles,'#styleLibrary','style')
for(const b of document.querySelectorAll('.nav-btn'))b.onclick=()=>{document.querySelectorAll('.nav-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));$('#'+b.dataset.view).classList.add('active');scrollTo({top:0,behavior:'smooth'})}
function updateGenerate(){const ready=$('#content').files.length&&$('#style').files.length;$('#generate').disabled=!ready;$('#generate span:first-child').textContent=ready?'Generate':'Choose content and style'}
function preview(input,title,img,drop){input.onchange=()=>{const file=input.files[0];if(!file)return;title.textContent=file.name;img.src=URL.createObjectURL(file);drop.classList.add('has-image');updateGenerate()}}
preview($('#content'),$('#contentTitle'),$('#contentPreview'),$('#contentDrop'))
preview($('#style'),$('#styleTitle'),$('#stylePreview'),$('#styleDrop'))
async function remoteToFile(url,name){const r=await fetch(url);if(!r.ok)throw Error('Could not load example');const blob=await r.blob();return new File([blob],name,{type:blob.type||'image/jpeg'})}
function setFile(input,file){const dt=new DataTransfer();dt.items.add(file);input.files=dt.files;input.dispatchEvent(new Event('change'))}
async function choose(item,type,index){const input=type==='content'?$('#content'):$('#style');const title=type==='content'?$('#contentTitle'):$('#styleTitle');const img=type==='content'?$('#contentPreview'):$('#stylePreview');const drop=type==='content'?$('#contentDrop'):$('#styleDrop');document.querySelectorAll(`.library-card[data-type="${type}"]`).forEach(x=>x.classList.remove('selected'));document.querySelector(`.library-card[data-type="${type}"][data-index="${index}"]`).classList.add('selected');try{const file=await remoteToFile(item.url,type+'.jpg');setFile(input,file);title.textContent=item.name;img.src=item.url;drop.classList.add('has-image');updateGenerate()}catch(e){alert('This reference needs internet access. Use your own image instead.')}}
document.addEventListener('click',e=>{const card=e.target.closest('.library-card');if(!card)return;const type=card.dataset.type;const index=Number(card.dataset.index);choose(type==='content'?contents[index]:styles[index],type,index)})
for(const b of document.querySelectorAll('.method'))b.onclick=()=>{document.querySelectorAll('.method').forEach(x=>x.classList.remove('active'));b.classList.add('active');method=b.dataset.method;$('#methodText').textContent={adain:'Transfer style through adaptive feature statistics.',gatys:'Optimize an image to match content and style representations.',johnson:'Apply a learned transformation network.'}[method];$('#alphaControl').classList.toggle('hidden',method!=='adain');$('#stepsControl').classList.toggle('hidden',method!=='gatys')}
$('#resolution').oninput=e=>$('#resolutionValue').textContent=e.target.value+' px'
$('#alpha').oninput=e=>$('#alphaValue').textContent=Math.round(e.target.value*100)+'%'
$('#steps').oninput=e=>$('#stepsValue').textContent=e.target.value
$('#generate').onclick=async()=>{const c=$('#content').files[0],s=$('#style').files[0];if(!c||!s)return;const btn=$('#generate');btn.disabled=true;btn.innerHTML='<span>Generating</span><span>· · ·</span>';const fd=new FormData();fd.append('content',c);fd.append('style',s);fd.append('method',method);fd.append('resolution',$('#resolution').value);fd.append('steps',$('#steps').value);fd.append('alpha',$('#alpha').value);try{const r=await fetch('/api/stylize',{method:'POST',body:fd});const d=await r.json();if(!r.ok)throw Error(d.error||'Generation failed');lastOriginal=URL.createObjectURL(c);lastResult=d.image;$('#resultImage').src=d.image;$('#resultMethod').textContent=method==='adain'?'AdaIN':method[0].toUpperCase()+method.slice(1);$('#metrics').innerHTML=`<span>${d.runtime_seconds}s</span><span>${d.resolution}px</span>${d.peak_gpu_memory_mb?`<span>${d.peak_gpu_memory_mb} MB VRAM</span>`:''}`;$('#result').classList.remove('hidden');$('#compare').classList.add('hidden');$('#result').scrollIntoView({behavior:'smooth'})}catch(e){alert(e.message)}finally{btn.disabled=false;btn.innerHTML='<span>Generate</span><span>→</span>';updateGenerate()}}
$('#compareBtn').onclick=()=>{if(!lastOriginal||!lastResult)return;$('#compareOriginal').src=lastOriginal;$('#compareResult').src=lastResult;$('#compare').classList.remove('hidden');$('#compare').scrollIntoView({behavior:'smooth'})}
$('#closeCompare').onclick=()=>$('#compare').classList.add('hidden')
$('#compareSlider').oninput=e=>$('.compare-after').style.width=e.target.value+'%'
$('#saveBtn').onclick=()=>{if(!lastResult)return;const a=document.createElement('a');a.href=lastResult;a.download='studio-nst-result.png';a.click()}
$('#newBtn').onclick=()=>{$('#result').classList.add('hidden');$('#compare').classList.add('hidden');scrollTo({top:0,behavior:'smooth'})}
$('#resetLibrary').onclick=()=>{for(const id of ['content','style'])$( '#'+id).value='';for(const id of ['contentDrop','styleDrop'])$('#'+id).classList.remove('has-image');$('#contentTitle').textContent='Upload your own';$('#styleTitle').textContent='Upload your own';document.querySelectorAll('.library-card').forEach(x=>x.classList.remove('selected'));$('#contentPreview').removeAttribute('src');$('#stylePreview').removeAttribute('src');updateGenerate()}
updateGenerate()
