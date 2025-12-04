// TMCRPrintPreview.jsx
// Fully remade header format — barangay left, section index right.

import React from "react";

function TMCRPrintPreview({
  provinceName,
  provinceIndex,
  municipalityName,
  municipalityIndex,
  barangayName,
  barangayIndex,
  sectionIndex,
  reportData,
  buttonClass = "tmcr-generate tmcr-col-btn"
}) {

  const openPrint = () => {
    if (!reportData || reportData.length === 0) return;

    const win = window.open("", "_blank", "width=1024,height=800");

    const style = `
      <style>
        body { font-family: Arial, sans-serif; padding: 28px; }
        h1 { margin: 0; font-size: 24px; text-align:center; letter-spacing:1px; }
        table {
          width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px;
        }
        th, td {
          border: 1px solid #9d9d9d; padding: 6px; text-align: left;
        }
        th { background: #e3e3e3; }
        tr:nth-child(even) { background: #fafafa; }
        .uline { display:inline-block; border-bottom:1px solid #000; padding:0 3px; }
      </style>
    `;

    const headerHtml = `
      <div style="text-align:center; margin-bottom:25px;">

        <!-- TITLE -->
        <h1><b>TAX MAP CONTROL ROLL</b></h1>

        <!-- PROVINCE + MUNICIPALITY (CENTERED BLOCK) -->
        <div style="margin-top:14px; font-size:16px; line-height:1.6;">
          <div>
            Prov./City:
            <span class="uline" style="min-width:150px;">${provinceName}</span>
            &nbsp; (Index No. <span class="uline" style="min-width:50px;">${provinceIndex}</span>)
          </div>
          <div>
            Mun./District:
            <span class="uline" style="min-width:150px;">${municipalityName}</span>
            &nbsp; (Index No. <span class="uline" style="min-width:50px;">${municipalityIndex}</span>)
          </div>
        </div>

        <!-- BARANGAY LEFT + SECTION RIGHT -->
        <div 
          style="
            margin-top:18px;
            font-size:16px;
            display:flex;
            justify-content:space-between;
            width:100%;
            padding:0 40px;
          "
        >

          <!-- LEFT SIDE -->
          <div style="text-align:left;">
            Barangay:
            <span class="uline" style="min-width:160px;">${barangayName}</span>
            &nbsp; (Index No. <span class="uline" style="min-width:50px;">${barangayIndex}</span>)
          </div>

          <!-- RIGHT SIDE -->
          <div style="text-align:right;">
            Section Index No.:
            <span class="uline" style="min-width:60px;">${sectionIndex}</span>
          </div>

        </div>

      </div>
    `;

    const rows = reportData
      .map(
        (r) => `
        <tr>
          <td>${r.sect_code || ""}</td>
          <td>${r.parcel_cod || ""}</td>
          <td>${r.lot_no || ""}</td>
          <td>${r.blk_no || ""}</td>
          <td>${r.tct_no || ""}</td>
          <td>${r.land_area || ""}</td>
          <td>${r.land_class || ""}</td>
          <td>${r.owner_name || ""}</td>
          <td>${r.land_arpn || ""}</td>
          <td>${r.bldg_class || ""}</td>
        </tr>
      `
      )
      .join("");

    win.document.write(`
      <html>
        <head>
          <title>TMCR Report – Print Preview</title>
          ${style}
        </head>
        <body>

          ${headerHtml}

          <table>
            <thead>
              <tr>
                <th>Section</th>
                <th>Parcel</th>
                <th>Lot</th>
                <th>Block</th>
                <th>TCT</th>
                <th>Area</th>
                <th>Classification</th>
                <th>Owner</th>
                <th>ARP No.</th>
                <th>BLDG</th>
              </tr>
            </thead>
            <tbody>
              ${rows}
            </tbody>
          </table>

          <script> window.onload = () => window.print(); </script>
        </body>
      </html>
    `);

    win.document.close();
  };

  return (
    <button
      className={buttonClass}
      disabled={!reportData?.length}
      onClick={openPrint}
    >
      Print Preview
    </button>
  );
}

export default TMCRPrintPreview;
