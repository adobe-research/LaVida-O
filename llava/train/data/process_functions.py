from llava.train.data.text_image import preproces_text_to_image_generation_s3,preproces_text_to_image_generation_piat_hash,preproces_text_to_image_generation_s3_loc,preproces_text_to_image_generation_layout_sam
from llava.train.data.und import preproces_und_mammoth_si_10M,preproces_grandf,preproces_refcoco_rec,preproces_vw_instruct,preprocess_reflection_und
from llava.train.data.interleaved import process_metaquery,process_uniworld,process_sharegpt_4o_edit,process_gpt_edit,process_gpt_edit_loc_gnd,process_pica_banana
PROCESS_FUNCTIONs = {
    "preproces_text_to_image_generation_s3": preproces_text_to_image_generation_s3,
     "preproces_text_to_image_generation_piat_hash": preproces_text_to_image_generation_piat_hash,
     'preproces_und_mammoth_si_10M':preproces_und_mammoth_si_10M,
     'preproces_grandf':preproces_grandf,
     'preproces_refcoco_rec':preproces_refcoco_rec,
     'preproces_vw_instruct':preproces_vw_instruct,
     'process_metaquery':process_metaquery,
     'process_uniworld':process_uniworld,
     'process_sharegpt_4o_edit':process_sharegpt_4o_edit,
     'process_gpt_edit':process_gpt_edit,
     'preprocess_reflection_und':preprocess_reflection_und,
     'preproces_text_to_image_generation_s3_loc':preproces_text_to_image_generation_s3_loc,
     'process_gpt_edit_loc_gnd': process_gpt_edit_loc_gnd,
     'preproces_text_to_image_generation_layout_sam':preproces_text_to_image_generation_layout_sam,
     'process_pica_banana':process_pica_banana
}