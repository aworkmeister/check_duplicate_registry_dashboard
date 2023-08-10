from redcap import Project
import os
import csv
import pandas as pd
from dash import dcc, html
from numpy import nan
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc

# Basic functions needed across all pages
# Pre-defined variables
VUMC_LOGO = 'https://www.vumc.org/marketing-engagement/sites/default/files/VUMC_Logo.jpg'
serviceurl = 'https://redcap.vanderbilt.edu/api/'
# Connect to all databases needed
map_tracking = Project(serviceurl, os.environ['VMAP_PARTICIPANT_TRACKING'])
edc = Project(serviceurl, os.environ['VMAP_ELECTRONIC_DATA_CAPTURE'])
np_dde = Project(serviceurl, os.environ['VMAP_NP_QX_DDE'])
vmap_qx = Project(serviceurl, os.environ['VMAP_QUESTIONNAIRES'])
elig_edc = Project(serviceurl, os.environ['VMAP_ELIGIBILITY_EDC'])
elig_np_dde = Project(serviceurl, os.environ['VMAP_ELIGIBILITY_DDE'])

# Timepoint specific information
timepoint_specific = {'enroll': {'epoch': '1', 'components': ['cmr_complete', 'echo_complete', 'np_complete',
                                                              'brain_complete', 'blood_complete', 'csf_complete'],
                                 'visit_complete_logic': "abpm_completed != '' & actigraphy_completed != '' "
                                                         "& cmr_complete != '' & echo_complete != '' & np_complete != ''  "
                                                         "& csf_complete != '' & brain_complete != '' & blood_complete != '' "
                                                         "& consent_date_time != ''",
                                 'dde_forms': ['neuropsych_administration_info', 'moca',
                                               'tower_coding_cw_hvot_cowat_tmt_bnt', 'biber_figure_learning_test',
                                               'symbol_span', 'digits_backwards']},
                      '9year': {'epoch': '6', 'components': ['cmr_complete', 'echo_complete', 'np_complete',
                                                             'brain_complete', 'blood_complete', 'csf_complete',
                                                             'int_ptp_complete', 'int_inf_complete'],
                                'visit_complete_logic': "abpm_completed != '' & actigraphy_completed != '' "
                                                        "& cmr_complete != '' & echo_complete != '' & np_complete != ''  "
                                                        "& csf_complete != '' & brain_complete != '' & blood_complete != '' "
                                                        "& consent_date_time != '' & int_inf_complete != '' "
                                                        "& int_ptp_complete != ''",
                                'dde_forms': ['neuropsych_administration_info', 'moca', 'symbol_span',
                                              'tower_coding_cw_hvot_cowat_tmt_bnt', 'biber_figure_learning_test',
                                              'digits_backwards']},
                      '18month': {'epoch': '2', 'components': ['cmr_complete', 'echo_complete', 'np_complete',
                                                               'brain_complete', 'blood_complete', 'csf_complete',
                                                               'int_ptp_complete', 'int_inf_complete'],
                                  'visit_complete_logic': "abpm_completed != '' & actigraphy_completed != '' "
                                                          "& cmr_complete != '' & echo_complete != '' & np_complete != ''  "
                                                          "& csf_complete != '' & brain_complete != '' & blood_complete != '' "
                                                          "& consent_date_time != '' & int_inf_complete != '' "
                                                          "& int_ptp_complete != ''",
                                  'dde_forms': ['neuropsych_administration_info', 'moca', 'symbol_span',
                                                'tower_coding_cw_hvot_cowat_tmt_bnt', 'biber_figure_learning_test',
                                                'digits_backwards']}
                      }


def get_data():
    """
    Get data from reports made on the three databases.
   :return:
    """
    # VMAP Participant Tracking Database
    all_ptd = map_tracking.export_reports(report_id='319315', format='df', df_kwargs={'dtype': 'str'})

    all_ptd_labels = map_tracking.export_records(
        fields=['map_id', 'visit_type', 'visit_resched_type', 'visit_resched2_type',
                'visit_resched3_type', 'epochs_missed_details'],
        raw_or_label='label', format='df', df_kwargs={'dtype': 'str'},
        export_checkbox_labels=True)
    all_ptd_labels = all_ptd_labels[~pd.isna(all_ptd_labels['map_id'])]
    # A crazy replacement
    all_ptd_labels['redcap_event_name'] = \
        all_ptd_labels['redcap_event_name'].str.replace('[^0-9a-zA-Z ]', '').str.lower().str.replace(' ',
                                                                                                     '_') + '_arm_1'
    all_ptd_labels = all_ptd_labels.set_index(['vmac_id', 'redcap_event_name'])
    all_ptd = all_ptd.set_index(['vmac_id', 'redcap_event_name'])
    # Checkbox columns get pulled with 0, replace with NaN for the update thing to work correctly
    checkbox_cols = [item for item in all_ptd.columns if '___' in item]
    all_ptd[checkbox_cols] = all_ptd[checkbox_cols].replace('0', nan)
    all_ptd.update(all_ptd_labels)
    all_ptd = all_ptd.reset_index()
    all_edc = edc.export_reports(report_id='299612', format='df', df_kwargs={'dtype': 'str'})
    all_edc_labels = edc.export_records(fields=['int_summ_exam', 'int_summ_review_team'], raw_or_label='label',
                                        format='df', df_kwargs={'dtype': 'str'})
    all_edc_labels['redcap_event_name'] = \
        all_edc_labels['redcap_event_name'].str.replace('[^0-9a-zA-Z ]', '').str.lower().str.replace(' ',
                                                                                                     '_') + '_arm_1'
    all_edc = all_edc.set_index(['map_id', 'redcap_event_name'])
    all_edc_labels = all_edc_labels.set_index(['map_id', 'redcap_event_name'])
    all_edc.update(all_edc_labels)
    all_edc = all_edc.reset_index()
    all_ptd.to_csv("ptd_data.csv", index=False)
    all_edc.to_csv("edc_data.csv", index=False)

    # VMAP NP EDC DDE
    np_dde_fields = ['neuropsych_administration_info_complete', 'moca_complete',
                     'tower_coding_cw_hvot_cowat_tmt_bnt_complete', 'biber_figure_learning_test_complete',
                     'symbol_span_complete', 'digits_backwards_complete', 'np_complete_date_time',
                     'np_chart_review']
    np_dde_data = np_dde.export_records(format='df', fields=np_dde_fields)
    np_dde_data = np_dde_data.reset_index(level=0)
    np_dde_data['np_chart_review'] = np_dde_data['np_chart_review'].fillna(0).astype(int)
    np_dde_labels = np_dde.export_records(format='df', fields=['np_examiner', 'np_scorer_secondary'],
                                          raw_or_label='label')
    np_dde_labels = np_dde_labels.reset_index(level=0)
    np_dde_data_final = np_dde_data.merge(np_dde_labels, on='record_id')
    np_dde_data_final.to_csv("np_dde_data.csv", index=False)

    # VMAP Qx data
    fem = vmap_qx.export_fem()
    events_for_qx = list(set([item['unique_event_name'] for item in fem if item['unique_event_name'] not in
                     ['eligibility_arm_1']]))
    forms = [item['form'] for item in fem if item['unique_event_name'] in events_for_qx]
    fields = [item['field_name'] for item in vmap_qx.metadata if item['form_name'] in forms and 'Did' in
              item['field_label'] and 'complete' in item['field_label']]
    qx_complete = vmap_qx.export_records(fields=fields + ['map_id', 'sex'] + ['abp_date_time', 'abp_review1_date_time',
                                                                              'abp_review2_date_time', 'actigraph_date_time',
                                                                              'actigraph_review1_date_time',
                                                                              'actigraph_review2_date_time'],
                                         events=events_for_qx, format='df',
                                         df_kwargs={'dtype': 'str'})
    # The medications and surgical history forms are repeating so we take care of that
    qx_complete = qx_complete.groupby(['vmac_id', 'redcap_event_name']).first().reset_index()
    review_fields = [item['field_name'] for item in vmap_qx.metadata if item['form_name'] in forms and
                     (item['field_name'].endswith('_review')) or (item['field_name'].endswith('_reviewed'))]
    qx_review = vmap_qx.export_records(fields=review_fields, events=events_for_qx, raw_or_label='label',
                                       export_checkbox_labels=True, format='df', df_kwargs={'dtype': 'str'})
    qx_review = qx_review.groupby(['vmac_id', 'redcap_event_name']).first().reset_index()
    # Combine checkbox fields
    for item in ['mhx', 'surg', 'metal', 'med']:
        field = '{}_review'.format(item)
        qx_review[field] = qx_review[['{}ed___1'.format(field), '{}ed___2'.format(field),
                                      '{}ed___3'.format(field)]].apply(lambda x: ','.join(x.dropna()), axis=1)
    # Drop individual checkbox fields
    qx_review = qx_review.loc[:, ~qx_review.columns.str.contains('___', case=False)]
    qx_review['redcap_event_name'] = \
        qx_review['redcap_event_name'].str.replace('[^0-9a-zA-Z ]', '').str.lower().str.replace(' ', '_') + '_arm_1'
    # Merge both qx df's to get a single data frame
    qx_data = qx_complete.merge(qx_review, on=['vmac_id', 'redcap_event_name'], how='inner')
    # Write everything out as CSV
    qx_data.to_csv("qx_data.csv", index=False)


def read_data(timepoint):
    # Create the page & read files
    if not os.path.isfile('ptd_data.csv'):
        get_data()
    ptd_data = pd.read_csv('ptd_data.csv', dtype='object')
    ptd_data = ptd_data[ptd_data['redcap_event_name'].str.contains(timepoint)]
    epochs_missed_details = [ptd_data.columns.get_loc(item) for item in ptd_data.columns if
                             item.startswith('epochs_missed_details___')]
    ptd_data['Missed Epochs'] = ptd_data[ptd_data.columns[epochs_missed_details]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    visit1_date_cols = [ptd_data.columns.get_loc(item) for item in ptd_data.columns if
                        'visit1' in item and item.endswith('date')]
    ptd_data['Day 1 Visit Date'] = ptd_data[ptd_data.columns[visit1_date_cols]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    visit_type_cols = [ptd_data.columns.get_loc(item) for item in ptd_data.columns if
                       'visit' in item and item.endswith('type')]
    ptd_data['Visit Type'] = ptd_data[ptd_data.columns[visit_type_cols]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)
    edc_data = pd.read_csv('edc_data.csv', dtype='object')
    edc_data = edc_data[edc_data['redcap_event_name'].str.contains(timepoint)]
    edc_data = edc_data.groupby(['map_id']).first().reset_index()
    edc_data = edc_data.fillna(value=nan)
    str_cols = ["map_id", "vmac_id", "np_examiner", "np_scorer_secondary", "np_chart_review"]
    np_dde_data = pd.read_csv('np_dde_data.csv', converters={i: str for i in str_cols})  # No dtype because we need the
    # form_complete to be numeric
    # Subset to time point and then remove the timepoint string from the record id so we can merge with other datasets
    np_dde_data = np_dde_data[np_dde_data['record_id'].str.lower().str.contains(timepoint)]
    np_dde_data = np_dde_data.rename({'record_id': 'map_id'}, axis=1)
    np_dde_data['map_id'] = [item.replace("_{}".format(timepoint), "") for item in np_dde_data['map_id'].str.lower()]
    # Read Qx data
    qx_data = pd.read_csv("qx_data.csv", converters={'vmac_id': str, 'map_id': str})
    qx_data = qx_data[qx_data['redcap_event_name'].str.contains(timepoint)]

    return [ptd_data, edc_data, np_dde_data, qx_data]


def get_elig_data():
    """
    Get data for eligibility
   :return:
    """
    # Eligibility EDC
    elig_edc_data = elig_edc.export_records(
        fields=['consent_date_time', 'elig_outcome', 'chart_finalization', 'chart_merge',
                'chart_finalization_review', 'vf_wrapup_date_time', 'np_complete'])
    # NP DDE data
    np_dde_data = elig_np_dde.export_records(fields=['np_complete_date_time', 'np_chart_review'])

    # Questionnaire Data
    fem = vmap_qx.export_fem()
    forms = [item['form'] for item in fem if item['unique_event_name'] == 'eligibility_arm_1']
    fields = [item['field_name'] for item in vmap_qx.metadata if item['form_name'] in forms and 'Did' in
              item['field_label'] and 'complete' in item['field_label']]
    qx_complete = vmap_qx.export_records(fields=fields + ['sex'], events=['eligibility_arm_1'], format='df',
                                         df_kwargs={'dtype': 'str'})
    qx_complete = qx_complete.groupby('vmac_id').first().reset_index()
    review_fields = [item['field_name'] for item in vmap_qx.metadata if item['form_name'] in forms and '_review' in
                     item['field_name']]
    qx_review = vmap_qx.export_records(fields=review_fields, events=['eligibility_arm_1'], raw_or_label='label',
                                       export_checkbox_labels=True, format='df', df_kwargs={'dtype': 'str'})
    qx_review = qx_review.groupby('vmac_id').first().reset_index()
    # Combine checkbox fields
    for item in ['mhx', 'surg', 'metal', 'med']:
        field = '{}_review'.format(item)
        qx_review[field] = qx_review[['{}ed___1'.format(field), '{}ed___2'.format(field),
                                      '{}ed___3'.format(field)]].apply(lambda x: ','.join(x.dropna()), axis=1)
    # Drop individual checkbox fields
    qx_review = qx_review.loc[:, ~qx_review.columns.str.contains('___', case=False)]
    # Merge both qx df's to get a single data frame
    qx_data = qx_complete.merge(qx_review, on='vmac_id', how='inner')
    qx_data['redcap_event_name'] = 'eligibility_arm_1'

    # Write everything out as CSV
    qx_data.to_csv("eligibility_qx_data.csv", index=False)
    with open('eligibility_edc_data.csv', 'w') as f:
        dict_writer = csv.DictWriter(f, elig_edc_data[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(elig_edc_data)
    with open('eligibility_dde_data.csv', 'w') as f:
        dict_writer = csv.DictWriter(f, np_dde_data[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(np_dde_data)
