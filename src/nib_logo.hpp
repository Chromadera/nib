#pragma once

namespace inkling {

// Full nib wordmark SVG (mirrored from docs/nib-logo.svg, transparent background)
inline constexpr const char * kNibLogoSvg = R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 680" width="1060" height="680" fill="none">
  <defs>
    <!-- Brushed cool titanium vertical gradient for the machined letter faces -->
    <linearGradient id="metalBody" x1="0" y1="230" x2="0" y2="640" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#eceef3"/>
      <stop offset="16%"  stop-color="#d4d9e2"/>
      <stop offset="30%"  stop-color="#bfc6d2"/>
      <stop offset="56%"  stop-color="#969ead"/>
      <stop offset="84%"  stop-color="#666e7e"/>
      <stop offset="96%"  stop-color="#596170"/>
      <stop offset="100%" stop-color="#6c7585"/>
    </linearGradient>

    <!-- Machined outer chamfer bevel (brighter top/sides, subtle bounce bottom) -->
    <linearGradient id="metalBevel" x1="0" y1="230" x2="0" y2="640" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.78"/>
      <stop offset="26%"  stop-color="#e3e7f0" stop-opacity="0.58"/>
      <stop offset="68%"  stop-color="#9ba3b2" stop-opacity="0.48"/>
      <stop offset="100%" stop-color="#b4bccb" stop-opacity="0.62"/>
    </linearGradient>

    <!-- Bright specular sheen on the upper ascender of 'b' right beneath the ferrule -->
    <linearGradient id="metalAscender" x1="635" y1="230" x2="706" y2="338" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#f5f7fb"/>
      <stop offset="55%"  stop-color="#d8dde7"/>
      <stop offset="100%" stop-color="#c2c9d5"/>
    </linearGradient>

    <!-- Ferrule / collar horizontal metallic sheen -->
    <linearGradient id="collarFill" x1="627.5" y1="192" x2="713.5" y2="220" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#c9cdd6"/>
      <stop offset="30%"  stop-color="#acafb9"/>
      <stop offset="72%"  stop-color="#838794"/>
      <stop offset="100%" stop-color="#737784"/>
    </linearGradient>

    <!-- Nib metallic outer shell / bevel -->
    <linearGradient id="nibRim" x1="624" y1="42" x2="717" y2="182" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#fbf3f5"/>
      <stop offset="32%"  stop-color="#e3cfd6"/>
      <stop offset="68%"  stop-color="#b49aa4"/>
      <stop offset="100%" stop-color="#877079"/>
    </linearGradient>

    <!-- Nib inner crimson-obsidian metallic core -->
    <radialGradient id="nibCore" cx="670.5" cy="136" r="78" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#cf2344" stop-opacity="0.96"/>
      <stop offset="36%"  stop-color="#921630" stop-opacity="0.96"/>
      <stop offset="72%"  stop-color="#4e1120" stop-opacity="0.97"/>
      <stop offset="100%" stop-color="#2c0c15" stop-opacity="0.98"/>
    </radialGradient>

    <!-- Top specular highlight on the upper nib tines -->
    <linearGradient id="nibTineSheen" x1="670.5" y1="42" x2="670.5" y2="126" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.90"/>
      <stop offset="44%"  stop-color="#f3d6df" stop-opacity="0.42"/>
      <stop offset="100%" stop-color="#cf2344" stop-opacity="0.0"/>
    </linearGradient>

    <!-- Soft crimson ambient glow behind the nib (clean on transparent background) -->
    <filter id="nibOuterGlow" x="-60%" y="-40%" width="220%" height="180%" color-interpolation-filters="sRGB">
      <feGaussianBlur in="SourceGraphic" stdDeviation="16" result="blurBig"/>
      <feGaussianBlur in="SourceGraphic" stdDeviation="5.5" result="blurTight"/>
      <feMerge>
        <feMergeNode in="blurBig"/>
        <feMergeNode in="blurTight"/>
      </feMerge>
    </filter>

    <!-- Crisp neon glow for the internal PCB circuit traces -->
    <filter id="traceGlow" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
      <feGaussianBlur in="SourceGraphic" stdDeviation="2.4" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- Exact letter paths (reused for clean inner chamfer bevel clipping) -->
    <path id="pathN" d="
      M 40 640
      L 40 468
      C 40 366, 118 295, 232 295
      C 346 295, 424 366, 424 468
      L 424 640
      L 353 640
      L 353 470
      C 353 402, 305 363, 232 363
      C 159 363, 111 402, 111 470
      L 111 640
      Z"/>

    <rect id="pathI" x="494" y="301" width="71" height="339"/>

    <path id="pathB" fill-rule="evenodd" d="
      M 635 230
      L 706 230
      L 706 345
      C 733 310, 777 295, 829 295
      C 944 295, 1019 363, 1019 467.5
      C 1019 572, 944 640, 827.5 640
      C 711 640, 635 572, 635 467.5
      Z
      M 827.5 363
      C 754 363, 706 404, 706 467.5
      C 706 531, 754 572, 827.5 572
      C 901 572, 949 531, 949 467.5
      C 949 404, 901 363, 827.5 363
      Z"/>

    <clipPath id="clipN">
      <path d="
        M 40 640 L 40 468
        C 40 366, 118 295, 232 295
        C 346 295, 424 366, 424 468
        L 424 640 L 353 640 L 353 470
        C 353 402, 305 363, 232 363
        C 159 363, 111 402, 111 470
        L 111 640 Z"/>
    </clipPath>
    <clipPath id="clipI">
      <rect x="494" y="301" width="71" height="339"/>
    </clipPath>
    <clipPath id="clipB">
      <path clip-rule="evenodd" d="
        M 635 230 L 706 230 L 706 345
        C 733 310, 777 295, 829 295
        C 944 295, 1019 363, 1019 467.5
        C 1019 572, 944 640, 827.5 640
        C 711 640, 635 572, 635 467.5 Z
        M 827.5 363
        C 754 363, 706 404, 706 467.5
        C 706 531, 754 572, 827.5 572
        C 901 572, 949 531, 949 467.5
        C 949 404, 901 363, 827.5 363 Z"/>
    </clipPath>

    <!-- Nib silhouette (left + right symmetric outline) -->
    <path id="nibOuterPath" d="
      M 668.0 42
      C 665.0 66, 648.5 98, 624.0 125
      C 634.5 143, 640.5 162, 642.5 182
      L 698.5 182
      C 700.5 162, 706.5 143, 717.0 125
      C 692.5 98, 676.0 66, 673.0 42
      Z"/>

    <!-- Inner recessed nib face for the dark crimson core & circuit traces -->
    <path id="nibInnerPath" d="
      M 667.2 58
      C 662.5 80, 648.5 104, 631.5 125
      C 640.5 141, 645.8 159, 647.8 179
      L 693.2 179
      C 695.2 159, 700.5 141, 709.5 125
      C 692.5 104, 678.5 80, 673.8 58
      Z"/>

    <!-- Transparent cutout mask: punches out the vertical slit and circular breather hole -->
    <mask id="nibCutoutMask" maskUnits="userSpaceOnUse" x="600" y="30" width="145" height="165">
      <rect x="600" y="30" width="145" height="165" fill="#ffffff"/>
      <!-- Vertical center slit from tip to breather hole -->
      <rect x="668.1" y="39" width="4.8" height="86" fill="#000000"/>
      <!-- Circular breather hole (aligned to cy=128) -->
      <circle cx="670.5" cy="128" r="12.4" fill="#000000"/>
    </mask>

    <!-- Clip for circuit traces so they stay strictly inside the inner nib face -->
    <clipPath id="nibInnerClip">
      <path d="
        M 667.2 58
        C 662.5 80, 648.5 104, 631.5 125
        C 640.5 141, 645.8 159, 647.8 179
        L 693.2 179
        C 695.2 159, 700.5 141, 709.5 125
        C 692.5 104, 678.5 80, 673.8 58 Z"/>
    </clipPath>
  </defs>

  <!-- ====================================================================== -->
  <!-- 1. LETTER "n"  (x: 40..424, y: 295..640, stroke: 71)                   -->
  <!-- ====================================================================== -->
  <g id="letter-n" clip-path="url(#clipN)">
    <use href="#pathN" fill="url(#metalBody)"/>
    <!-- Inner machined chamfer bevel -->
    <use href="#pathN" stroke="url(#metalBevel)" stroke-width="11" stroke-linejoin="miter"/>
    <use href="#pathN" stroke="#6f7787" stroke-opacity="0.55" stroke-width="2.5" stroke-linejoin="miter"/>
  </g>

  <!-- ====================================================================== -->
  <!-- 2. LETTER "i" — DOTLESS  (x: 494..565, y: 301..640, width: 71)         -->
  <!-- ====================================================================== -->
  <g id="letter-i" clip-path="url(#clipI)">
    <use href="#pathI" fill="url(#metalBody)"/>
    <!-- Inner machined chamfer bevel -->
    <use href="#pathI" stroke="url(#metalBevel)" stroke-width="11"/>
    <use href="#pathI" stroke="#6f7787" stroke-opacity="0.55" stroke-width="2.5"/>
  </g>

  <!-- ====================================================================== -->
  <!-- 3. LETTER "b"  (stem x: 635..706, bowl x: 635..1019, y: 230..640)      -->
  <!-- ====================================================================== -->
  <g id="letter-b" clip-path="url(#clipB)">
    <use href="#pathB" fill="url(#metalBody)"/>
    <!-- Bright specular sheen on the upper ascender stem right under the collar -->
    <rect x="635" y="230" width="71" height="84" fill="url(#metalAscender)" opacity="0.52"/>
    <!-- Inner machined chamfer bevel -->
    <use href="#pathB" stroke="url(#metalBevel)" stroke-width="11" stroke-linejoin="miter"/>
    <use href="#pathB" stroke="#6f7787" stroke-opacity="0.55" stroke-width="2.5" stroke-linejoin="miter"/>
  </g>

  <!-- ====================================================================== -->
  <!-- 4. FERRULE / COLLAR  (cx: 670.5, x: 627.5..713.5, y: 192..220)         -->
  <!-- ====================================================================== -->
  <g id="nib-collar">
    <rect x="627.5" y="192" width="86" height="28" rx="2.5" ry="2.5"
      fill="url(#collarFill)"
      stroke="#b7bcc8"
      stroke-opacity="0.55"
      stroke-width="1.5"/>
    <line x1="630" y1="194" x2="711" y2="194"
      stroke="#ffffff"
      stroke-opacity="0.48"
      stroke-width="1.5"/>
    <line x1="630" y1="218.5" x2="711" y2="218.5"
      stroke="#363942"
      stroke-opacity="0.55"
      stroke-width="1.5"/>
  </g>

  <!-- ====================================================================== -->
  <!-- 5. FOUNTAIN PEN NIB WITH GLOWING CRIMSON CIRCUIT TRACES (cx: 670.5)    -->
  <!-- ====================================================================== -->
  <g id="fountain-nib">
    <!-- Soft crimson outer halo (clean alpha fade on transparent background) -->
    <g filter="url(#nibOuterGlow)" opacity="0.72">
      <use href="#nibOuterPath" fill="#e61c40"/>
    </g>

    <!-- Main nib body with transparent slit + breather hole punched out -->
    <g mask="url(#nibCutoutMask)">
      <!-- Metallic outer bevel frame -->
      <use href="#nibOuterPath" fill="url(#nibRim)"/>

      <!-- Recessed crimson-obsidian core -->
      <use href="#nibInnerPath" fill="url(#nibCore)"/>

      <!-- Upper tine metallic sheen -->
      <use href="#nibOuterPath" fill="url(#nibTineSheen)"/>

      <!-- Symmetrical neon-red PCB circuit traces clipped inside the nib core -->
      <g clip-path="url(#nibInnerClip)" filter="url(#traceGlow)">
        <!-- Center subtle crimson spine glow below breather hole -->
        <line x1="670.5" y1="141" x2="670.5" y2="181"
          stroke="#ff4d6d" stroke-opacity="0.45" stroke-width="2.0" stroke-linecap="round"/>

        <!-- Left main circuit trace -->
        <path d="M 661.5 181 L 661.5 153 L 646.0 138 L 646.0 112"
          stroke="#ff4b68" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 661.5 181 L 661.5 153 L 646.0 138 L 646.0 112"
          stroke="#ffd1da" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Left upper pad / via -->
        <circle cx="646.0" cy="109" r="3.6" fill="#4a0d1c" stroke="#ff5c78" stroke-width="2.0"/>
        <circle cx="646.0" cy="109" r="1.3" fill="#ffe0e6"/>
        <!-- Left lower side-branch & pad -->
        <path d="M 661.5 161 L 653.5 161"
          stroke="#ff4b68" stroke-width="2.0" stroke-linecap="round"/>
        <circle cx="650.5" cy="161" r="2.8" fill="#4a0d1c" stroke="#ff5c78" stroke-width="1.7"/>

        <!-- Right main circuit trace (exact mirror) -->
        <path d="M 679.5 181 L 679.5 153 L 695.0 138 L 695.0 112"
          stroke="#ff4b68" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 679.5 181 L 679.5 153 L 695.0 138 L 695.0 112"
          stroke="#ffd1da" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Right upper pad / via -->
        <circle cx="695.0" cy="109" r="3.6" fill="#4a0d1c" stroke="#ff5c78" stroke-width="2.0"/>
        <circle cx="695.0" cy="109" r="1.3" fill="#ffe0e6"/>
        <!-- Right lower side-branch & pad -->
        <path d="M 679.5 161 L 687.5 161"
          stroke="#ff4b68" stroke-width="2.0" stroke-linecap="round"/>
        <circle cx="690.5" cy="161" r="2.8" fill="#4a0d1c" stroke="#ff5c78" stroke-width="1.7"/>
      </g>

      <!-- Crisp inner metallic bevel ring around breather hole -->
      <circle cx="670.5" cy="128" r="14.1"
        stroke="#ebd4dc" stroke-opacity="0.72" stroke-width="2.2"/>

      <!-- Crisp bright edges lining the center slit -->
      <line x1="667.4" y1="44" x2="667.4" y2="117"
        stroke="#f7e8ed" stroke-opacity="0.78" stroke-width="1.6"/>
      <line x1="673.6" y1="44" x2="673.6" y2="117"
        stroke="#f7e8ed" stroke-opacity="0.78" stroke-width="1.6"/>

      <!-- Crisp outer metallic rim stroke -->
      <use href="#nibOuterPath"
        stroke="#f5e6ec" stroke-opacity="0.72" stroke-width="1.8" stroke-linejoin="miter"/>
    </g>
  </g>
</svg>)SVG";

// Compact square nib + ferrule mark for browser favicon and collapsed sidebar
inline constexpr const char * kNibFaviconSvg = R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="563.5 24 214 214" width="64" height="64" fill="none">
  <defs>
    <linearGradient id="collarFill" x1="627.5" y1="192" x2="713.5" y2="220" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#c9cdd6"/>
      <stop offset="30%"  stop-color="#acafb9"/>
      <stop offset="72%"  stop-color="#838794"/>
      <stop offset="100%" stop-color="#737784"/>
    </linearGradient>
    <linearGradient id="nibRim" x1="624" y1="42" x2="717" y2="182" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#fbf3f5"/>
      <stop offset="32%"  stop-color="#e3cfd6"/>
      <stop offset="68%"  stop-color="#b49aa4"/>
      <stop offset="100%" stop-color="#877079"/>
    </linearGradient>
    <radialGradient id="nibCore" cx="670.5" cy="136" r="78" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#cf2344" stop-opacity="0.96"/>
      <stop offset="36%"  stop-color="#921630" stop-opacity="0.96"/>
      <stop offset="72%"  stop-color="#4e1120" stop-opacity="0.97"/>
      <stop offset="100%" stop-color="#2c0c15" stop-opacity="0.98"/>
    </radialGradient>
    <linearGradient id="nibTineSheen" x1="670.5" y1="42" x2="670.5" y2="126" gradientUnits="userSpaceOnUse">
      <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.90"/>
      <stop offset="44%"  stop-color="#f3d6df" stop-opacity="0.42"/>
      <stop offset="100%" stop-color="#cf2344" stop-opacity="0.0"/>
    </linearGradient>
    <filter id="nibOuterGlow" x="-60%" y="-40%" width="220%" height="180%" color-interpolation-filters="sRGB">
      <feGaussianBlur in="SourceGraphic" stdDeviation="14" result="blurBig"/>
      <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blurTight"/>
      <feMerge>
        <feMergeNode in="blurBig"/>
        <feMergeNode in="blurTight"/>
      </feMerge>
    </filter>
    <filter id="traceGlow" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">
      <feGaussianBlur in="SourceGraphic" stdDeviation="2.4" result="b"/>
      <feMerge>
        <feMergeNode in="b"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <path id="nibOuterPath" d="
      M 668.0 42
      C 665.0 66, 648.5 98, 624.0 125
      C 634.5 143, 640.5 162, 642.5 182
      L 698.5 182
      C 700.5 162, 706.5 143, 717.0 125
      C 692.5 98, 676.0 66, 673.0 42
      Z"/>
    <path id="nibInnerPath" d="
      M 667.2 58
      C 662.5 80, 648.5 104, 631.5 125
      C 640.5 141, 645.8 159, 647.8 179
      L 693.2 179
      C 695.2 159, 700.5 141, 709.5 125
      C 692.5 104, 678.5 80, 673.8 58
      Z"/>
    <mask id="nibCutoutMask" maskUnits="userSpaceOnUse" x="600" y="30" width="145" height="165">
      <rect x="600" y="30" width="145" height="165" fill="#ffffff"/>
      <rect x="668.1" y="39" width="4.8" height="86" fill="#000000"/>
      <circle cx="670.5" cy="128" r="12.4" fill="#000000"/>
    </mask>
    <clipPath id="nibInnerClip">
      <path d="
        M 667.2 58
        C 662.5 80, 648.5 104, 631.5 125
        C 640.5 141, 645.8 159, 647.8 179
        L 693.2 179
        C 695.2 159, 700.5 141, 709.5 125
        C 692.5 104, 678.5 80, 673.8 58 Z"/>
    </clipPath>
  </defs>
  <g id="nib-collar">
    <rect x="627.5" y="192" width="86" height="28" rx="2.5" ry="2.5"
      fill="url(#collarFill)" stroke="#b7bcc8" stroke-opacity="0.55" stroke-width="1.5"/>
    <line x1="630" y1="194" x2="711" y2="194" stroke="#ffffff" stroke-opacity="0.48" stroke-width="1.5"/>
    <line x1="630" y1="218.5" x2="711" y2="218.5" stroke="#363942" stroke-opacity="0.55" stroke-width="1.5"/>
  </g>
  <g id="fountain-nib">
    <g filter="url(#nibOuterGlow)" opacity="0.78">
      <use href="#nibOuterPath" fill="#e61c40"/>
    </g>
    <g mask="url(#nibCutoutMask)">
      <use href="#nibOuterPath" fill="url(#nibRim)"/>
      <use href="#nibInnerPath" fill="url(#nibCore)"/>
      <use href="#nibOuterPath" fill="url(#nibTineSheen)"/>
      <g clip-path="url(#nibInnerClip)" filter="url(#traceGlow)">
        <line x1="670.5" y1="141" x2="670.5" y2="181" stroke="#ff4d6d" stroke-opacity="0.45" stroke-width="2.0" stroke-linecap="round"/>
        <path d="M 661.5 181 L 661.5 153 L 646.0 138 L 646.0 112" stroke="#ff4b68" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 661.5 181 L 661.5 153 L 646.0 138 L 646.0 112" stroke="#ffd1da" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="646.0" cy="109" r="3.6" fill="#4a0d1c" stroke="#ff5c78" stroke-width="2.0"/>
        <circle cx="646.0" cy="109" r="1.3" fill="#ffe0e6"/>
        <path d="M 661.5 161 L 653.5 161" stroke="#ff4b68" stroke-width="2.0" stroke-linecap="round"/>
        <circle cx="650.5" cy="161" r="2.8" fill="#4a0d1c" stroke="#ff5c78" stroke-width="1.7"/>
        <path d="M 679.5 181 L 679.5 153 L 695.0 138 L 695.0 112" stroke="#ff4b68" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M 679.5 181 L 679.5 153 L 695.0 138 L 695.0 112" stroke="#ffd1da" stroke-width="1.0" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="695.0" cy="109" r="3.6" fill="#4a0d1c" stroke="#ff5c78" stroke-width="2.0"/>
        <circle cx="695.0" cy="109" r="1.3" fill="#ffe0e6"/>
        <path d="M 679.5 161 L 687.5 161" stroke="#ff4b68" stroke-width="2.0" stroke-linecap="round"/>
        <circle cx="690.5" cy="161" r="2.8" fill="#4a0d1c" stroke="#ff5c78" stroke-width="1.7"/>
      </g>
      <circle cx="670.5" cy="128" r="14.1" stroke="#ebd4dc" stroke-opacity="0.72" stroke-width="2.2"/>
      <line x1="667.4" y1="44" x2="667.4" y2="117" stroke="#f7e8ed" stroke-opacity="0.78" stroke-width="1.6"/>
      <line x1="673.6" y1="44" x2="673.6" y2="117" stroke="#f7e8ed" stroke-opacity="0.78" stroke-width="1.6"/>
      <use href="#nibOuterPath" stroke="#f5e6ec" stroke-opacity="0.72" stroke-width="1.8" stroke-linejoin="miter"/>
    </g>
  </g>
</svg>)SVG";

} // namespace inkling
