import { AutoModelForCausalLM, AutoTokenizer, Tensor } from './lib.js';
let model_id = 'Murden/polyglot-ko-qabot';
let tokenizer = await AutoTokenizer.from_pretrained(model_id);
var model;


var max_length = 80;
var inp_leng;
var inp_container = new BigInt64Array(max_length);
var attn_container = new BigInt64Array(max_length);


function callback(data, type){
    if (type == 'generate') {
        data = {
            output_token_ids: data[0].output_token_ids,
            maxNumToken: (max_length - inp_leng)
        }
    }
    let msg = {
        type: type,
        data: data
    };
    self.postMessage(msg);
}


self.addEventListener('message', async (event) => {
  const message = event.data;

  switch (message.type) {
    case 'model':
        if (!model){
          model = await AutoModelForCausalLM.from_pretrained(model_id, 
            {progress_callback: data => callback(data, 'model')})
        }
        break;
    case 'generate':
        let prompt = message.prompt;
        let {input_ids, attention_mask} = await tokenizer(prompt);

        inp_leng = input_ids.data.length;
        inp_container.fill(0n);
        attn_container.fill(0n);
        inp_container.set(input_ids.data, 0);
        attn_container.set(attention_mask.data, 0);
        
        let inp_ids = new Tensor('int64', inp_container, [1, inp_container.length]);
        let attn_mask = new Tensor('int64', attn_container, [1, attn_container.length]);
        let gen_config = {max_length: max_length-inp_leng, callback_function: data => callback(data, 'generate')};

        let outputs = await model.generate(inp_ids, inp_leng, gen_config, null, {inputs_attention_mask: attn_mask});
        let decoded = await tokenizer.decode(outputs[0], { skip_special_tokens: true });

        callback({decoded: decoded}, 'output');
        break;
  }
})